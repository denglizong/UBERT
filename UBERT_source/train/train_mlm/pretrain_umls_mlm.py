import sys
import warnings
import pickle as pk
import transformers
from absl import flags, logging, app
from pathlib import Path
from transformers import BertTokenizer
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import LineByLineTextDatasetforMLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import set_seed
import torch 
import logging
logging.basicConfig(filename='mlm_INFO.log', level=logging.INFO)
torch.cuda.empty_cache()
warnings.simplefilter(action='ignore', category=FutureWarning)
transformers.logging.set_verbosity_info()

FLAGS = flags.FLAGS

flags.DEFINE_string("vocab_file", None, "Vocab file generated from the tokenizer pretraining")
flags.DEFINE_string("input_data", None, "A single txt file of pubmed+pmc+umls atom strings data")
flags.DEFINE_string("output_dir", None, "Directory to save the trained output")
flags.DEFINE_string("trained_model_dir", None, "Directory to save the trained model")
flags.DEFINE_string("main_log_file", "./main_mlm.log", "Log file location with name")
flags.DEFINE_integer("block_size", 32, "Maximum data block size to be processed, in UMLS only MLM situatino our block size should be size of aui string size")
flags.DEFINE_integer("num_train_epochs", None, "Number of training epochs")
flags.DEFINE_integer("per_device_train_batch_size", None, "Trining batch size; 128 is compatible with university gpus")
#flags.DEFINE_bool("resume_checkpoint", False, "Resume training checkpoint (True or False)")

flags.DEFINE_integer("vocab_size", 50000, "Vocabulary size for BertConfig")
flags.DEFINE_integer("max_position_embeddings", 512, "max_position_embeddings for BertConfig")
flags.DEFINE_string("logging_strategy", "steps", "Log output info stepwise")
flags.DEFINE_integer("logging_steps", 500, "Log output info every 500 steps")
flags.DEFINE_string("save_strategy", "steps", "Save output info stepwise")
flags.DEFINE_integer("save_steps", 500, "Save output every 500 steps")
#flags.DEFINE_integer("save_total_limit", 2, "")
flags.DEFINE_bool("prediction_loss_only", False, "When performing evaluation and predictions, only return the loss")
flags.DEFINE_float("mlm_probability", 0.15, "Portion of words to be masked; higher the portion harder for the model to learn")
flags.DEFINE_integer("local_rank", 0, "Rank of the process during distributed training")

def main(_):
    sys.stdout=open(FLAGS.main_log_file,"w")

    #set_seed(42)
    #device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Start fitting tokenizer...")
    tokenizer = BertTokenizer(FLAGS.vocab_file)

    config = BertConfig(
        vocab_size=FLAGS.vocab_size,
        max_position_embeddings=FLAGS.max_position_embeddings,
    )

    print("Start training from scratch ...")
    # training from scratch
    model = BertForMaskedLM(config=config)

    data_collator = DataCollatorForLanguageModeling(
			tokenizer=tokenizer,
			mlm=True,
			mlm_probability=FLAGS.mlm_probability)

    training_args=TrainingArguments(
		output_dir=FLAGS.output_dir,
		overwrite_output_dir=True,
		num_train_epochs=FLAGS.num_train_epochs,
		per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        logging_strategy=FLAGS.logging_strategy,
        logging_steps=FLAGS.logging_steps,
        save_strategy=FLAGS.save_strategy,
		save_steps=FLAGS.save_steps,
		#save_total_limit=FLAGS.save_total_limit,
        #resume_from_checkpoint=FLAGS.resume_checkpoint,
		prediction_loss_only=FLAGS.prediction_loss_only,
        fp16=True,
        sharded_ddp='simple',
        local_rank=FLAGS.local_rank,
	)

    # loading dataset from pickle
    print("loading dataset...")
    #dataset = pk.load( open(FLAGS.input_data_file, "rb") )
    dataset = LineByLineTextDatasetforMLM(
        tokenizer=tokenizer,
        file_path=FLAGS.input_data,
        block_size=FLAGS.block_size,
    )
    print("finished loading dataset...")

    trainer = Trainer(
		model=model,
		args=training_args,
		data_collator=data_collator,
		train_dataset=dataset
	)

    # Starting training
    print("Start training...")
    #trainer.train()
    trainer.train()
    
    print("Saving model to %s"%FLAGS.trained_model_dir)
    trainer.save_model(FLAGS.trained_model_dir)
    
    sys.stdout.close()

if __name__=="__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("input_data")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("trained_model_dir")
    flags.mark_flag_as_required("num_train_epochs")
    flags.mark_flag_as_required("per_device_train_batch_size")
    app.run(main)
