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
logging.basicConfig(filename='mlm_sp_INFO.log', level=logging.INFO)
torch.cuda.empty_cache()
warnings.simplefilter(action='ignore', category=FutureWarning)
transformers.logging.set_verbosity_info()

FLAGS = flags.FLAGS

flags.DEFINE_string("vocab_file", None, "Vocab file generated from the tokenizer pretraining")
flags.DEFINE_string("aui_vec_file_path", None, ".pkl file with aui anc corresponding vector")
flags.DEFINE_string("train_data", None, "Train datafile")
flags.DEFINE_string("eval_data", None, "Eval datafile")
flags.DEFINE_string("out_dir", None, "Directory to save checkpoints")
flags.DEFINE_string("trained_model_dir", None, "Directory to save the trained model")
flags.DEFINE_integer("num_train_epochs", None, "Number of epochs to train for")
flags.DEFINE_integer("per_device_train_batch_size", None, "Trining batch size")
flags.DEFINE_integer("per_device_eval_batch_size", None, "Eval batch size")
flags.DEFINE_string("pretrained_model", None, "Path to the trained model to start this session of training from; in our case this is the trained MLM model")

flags.DEFINE_string("main_method_log","./main_method_mlm_sp.log", "System output will be saved here")
flags.DEFINE_string("logging_strategy", "epoch", "Log output info every epoch")
flags.DEFINE_string("save_strategy", "epoch", "Save output info every epoch")
flags.DEFINE_string("evaluation_strategy", "epoch", "Do evaluation every epoch")
flags.DEFINE_bool("load_best_model_at_end", True, "Whether to load the best model at end")
flags.DEFINE_integer("seed", 42, "Set seed")
flags.DEFINE_integer("vocab_size", 50000, "Size of the vocab")
flags.DEFINE_bool("overwrite_output_dir", True, "Whether to overwrite the output directory")
flags.DEFINE_bool("prediction_loss_only", False, "When performing evaluation and predictions, only return the loss")
flags.DEFINE_integer("local_rank", 0, "Rank of the process during distributed training")

def compute_metrics(p):
    pred, true_labels = p
    preds = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, pos_label=0, average='binary')
    accuracy = accuracy_score(y_true=true_labels, y_pred=preds)
    fpr, tpr, thresholds = roc_curve(true_labels, pred[:, 0], pos_label=0)
    auc_score = auc(fpr, tpr)
    #auc = roc_auc_score(labels, np.max(pred, axis=1))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc_score}

def main(_):
    sys.stdout=open(FLAGS.main_method_log, "w")
    tokenizer = BertTokenizer(FLAGS.vocab_file)
    train_dataset = StreamDatasetforSynonymyPrediction(
                    tokenizer=tokenizer,
                    aui_vec_file_path=FLAGS.aui_vec_file_path,
                    data_file_path=FLAGS.train_data,
    )
    eval_dataset = StreamDatasetforSynonymyPrediction(
                    tokenizer=tokenizer,
                    aui_vec_file_path=FLAGS.aui_vec_file_path,
                    data_file_path=FLAGS.eval_data,
    ) 
    #config = BertConfig(vocab_size=FLAGS.vocab_size)

    model = BertForNextSentencePrediction.from_pretrained(FLAGS.pretrained_model)

    data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
    )

    training_args=TrainingArguments(
            output_dir=FLAGS.out_dir,
            overwrite_output_dir=FLAGS.overwrite_output_dir,
            num_train_epochs=FLAGS.num_train_epochs,
            per_device_train_batch_size=FLAGS.per_device_train_batch_size,
            per_device_eval_batch_size=FLAGS.per_device_eval_batch_size,
            logging_strategy=FLAGS.logging_strategy,
            save_strategy=FLAGS.save_strategy,
            evaluation_strategy=FLAGS.evaluation_strategy,
            load_best_model_at_end=FLAGS.load_best_model_at_end,
            prediction_loss_only=FLAGS.prediction_loss_only,
            seed=FLAGS.seed,
            fp16=True,
            sharded_ddp='simple',
            local_rank=FLAGS.local_rank,
        )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        
    print(f"Starting training...")
    trainer.train()
    print(f"Training finished...")

    save_model_loc = FLAGS.trained_model_dir
    print(f"Saving model to: {save_model_loc}")
    trainer.save_model(save_model_loc)
    print(f"Model saved...")

    sys.stdout.close()

if __name__=="__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("aui_vec_file_path")
    flags.mark_flag_as_required("train_data")
    flags.mark_flag_as_required("eval_data")
    flags.mark_flag_as_required("out_dir")
    flags.mark_flag_as_required("trained_model_dir")
    flags.mark_flag_as_required("num_train_epochs")
    flags.mark_flag_as_required("per_device_train_batch_size")
    flags.mark_flag_as_required("per_device_eval_batch_size")
    flags.mark_flag_as_required("pretrained_model")
    
    app.run(main)
