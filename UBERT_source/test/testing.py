import sys
import torch
import warnings
import pickle as pk
import numpy as np
from absl import flags, logging, app
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
import transformers
from transformers import set_seed
from transformers import StreamDatasetforSynonymyPrediction
from transformers import BertConfig, BertTokenizer, BertForNextSentencePrediction
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import logging
logging.basicConfig(filename='transformer_INFO.log', level=logging.INFO)
torch.cuda.empty_cache()
warnings.simplefilter(action='ignore', category=FutureWarning)
transformers.logging.set_verbosity_info()

FLAGS = flags.FLAGS

flags.DEFINE_string("vocab_file", None, "Vocab file generated from the tokenizer pretraining")
flags.DEFINE_string("aui_vec_file_path", None, ".pkl file with aui anc corresponding vector")
flags.DEFINE_string("test_data", None, "Test datafile")
flags.DEFINE_string("out_dir", None, "Directory to save test output")
flags.DEFINE_integer("per_device_eval_batch_size", None, "Eval batch size; 128 is compatible with RCI gpus")
flags.DEFINE_string("checkpoint_model", None, "Path to the trained model to start this session of training from; in this case it is the checkpoints from already trained SP model")

flags.DEFINE_string("main_method_log", None, "System output will be saved here")
flags.DEFINE_string("checkpoint_name", None, "Checkpoint name will be passed to use in labels and predictions file names")
flags.DEFINE_string("logging_strategy", "epoch", "Log output info every epoch")
flags.DEFINE_string("save_strategy", "epoch", "Save output info every epoch")
flags.DEFINE_string("evaluation_strategy", "epoch", "Do evaluation every epoch")
flags.DEFINE_bool("overwrite_output_dir", True, "Whether to overwrite the output directory")
flags.DEFINE_integer("local_rank", 0, "Rank of the process during distributed training")

def compute_metrics(p):
    # TODO: Log metrics data in a better organized manner
    pred, true_labels = p
    preds = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, pos_label=0,  average='binary')
    accuracy = accuracy_score(y_true=true_labels, y_pred=preds)
    fpr, tpr, thresholds = roc_curve(true_labels, pred[:, 0], pos_label=0)
    auc_score = auc(fpr, tpr)
    with open("./correct_metrics_labels_"+FLAGS.checkpoint_name+".txt", "w") as lwf:
        np.savetxt(lwf, true_labels, fmt='%d')   
    with open("./correct_metrics_predictions_"+FLAGS.checkpoint_name+".txt", "w") as pwf:
        np.savetxt(pwf, pred, fmt='%1.5f')
    print(f"Metrics for checkpoint: {FLAGS.checkpoint_model}")
    print(f"Accuracy: {accuracy} | F1: {f1} | Precision: {precision} | Recall: {recall} |AUC: {auc_score}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc_score}

def main(_):
    sys.stdout=open(FLAGS.main_method_log, "w")
    tokenizer = BertTokenizer(FLAGS.vocab_file)
    test_dataset = StreamDatasetforSynonymyPrediction(
                    tokenizer=tokenizer,
                    aui_vec_file_path=FLAGS.aui_vec_file_path,
                    data_file_path=FLAGS.test_data,
    )
    model = BertForNextSentencePrediction.from_pretrained(FLAGS.checkpoint_model, local_files_only=True)

    data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
    )

    training_args=TrainingArguments(
            output_dir=FLAGS.out_dir,
            overwrite_output_dir=FLAGS.overwrite_output_dir,
            per_device_eval_batch_size=FLAGS.per_device_eval_batch_size,
            logging_strategy=FLAGS.logging_strategy,
            save_strategy=FLAGS.save_strategy,
            evaluation_strategy=FLAGS.evaluation_strategy,
            do_predict=True,
            fp16=True,
            sharded_ddp='simple',
            local_rank=FLAGS.local_rank,
        )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
    print(f"Starting predictions...")
    trainer.predict(test_dataset)
    print(f"Predictions finished...")

    sys.stdout.close()

if __name__=="__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("aui_vec_file_path")
    flags.mark_flag_as_required("test_data")
    flags.mark_flag_as_required("out_dir")
    flags.mark_flag_as_required("per_device_eval_batch_size")
    flags.mark_flag_as_required("checkpoint_model")
    flags.mark_flag_as_required("main_method_log")
    flags.mark_flag_as_required("checkpoint_name")
    
    app.run(main)
