# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_lm_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should look for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

# xxxx added for streaming data to MLM task
class LineByLineTextDatasetforMLM(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    #print(f"Inside LineByLineTextDatasetforMLM")
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")
                
        self.tokenizer = tokenizer        
        self.input_file = file_path
        self.process_block_size = block_size

        count = 0
        with open(self.input_file, encoding="utf-8") as lf:
            for ind, l in enumerate(lf):
                count = count + 1
        #print(f"file length {count}")
        self.file_length = count

        #line_num = 102
        #with open(file_path, encoding="utf-8") as testf:
        #    [next(testf) for x in range(line_num-1)]
        #    rel_line = next(testf)
        #print(f"{rel_line}")

        #with open(file_path, encoding="utf-8") as f:
            #lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        #print(f"First few lines:\n")
        #print(lines[3:])
        #lines = lines[10003]
        #batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        #print(f"batch_encoding:\n")
        #print(batch_encoding)
        #self.examples = batch_encoding["input_ids"]
        #self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        #print(f"inside def __len__(self)")
        return self.file_length
        #return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        print(f"{i}")
        with open(self.input_file, encoding="utf-8") as testf:
            [next(testf) for x in range(i-1)]
            rel_line = next(testf)
        rel_line = rel_line.strip("\n")
        batch_encoding = self.tokenizer(rel_line, add_special_tokens=True, truncation=True, max_length=self.process_block_size)
        e = batch_encoding["input_ids"]
        example = {"input_ids": torch.tensor(e, dtype=torch.long)}
        return example
        #return self.examples[i]

class LineByLineWithRefDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, ref_path: str):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm_wwm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        assert os.path.isfile(ref_path), f"Ref file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")
        logger.info(f"Use ref segment results at {ref_path}")
        with open(file_path, encoding="utf-8") as f:
            data = f.readlines()  # use this method to avoid delimiter '\u2029' to split a line
        data = [line.strip() for line in data if len(line) > 0 and not line.isspace()]
        # Get ref inf from file
        with open(ref_path, encoding="utf-8") as f:
            ref = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        assert len(data) == len(ref)

        batch_encoding = tokenizer(data, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

        n = len(self.examples)
        for i in range(n):
            self.examples[i]["chinese_ref"] = torch.tensor(ref[i], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isdir(file_dir)
        logger.info(f"Creating features from dataset file folder at {file_dir}")
        self.examples = []
        # TODO: randomness could apply a random seed, ex. rng = random.Random(random_seed)
        # file path looks like ./dataset/wiki_1, ./dataset/wiki_2
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            assert os.path.isfile(file_path)
            article_open = False
            with open(file_path, encoding="utf-8") as f:
                original_lines = f.readlines()
                article_lines = []
                for line in original_lines:
                    if "<doc id=" in line:
                        article_open = True
                    elif "</doc>" in line:
                        article_open = False
                        document = [
                            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
                            for line in article_lines[1:]
                            if (len(line) > 0 and not line.isspace())
                        ]

                        examples = self.create_examples_from_document(document, block_size, tokenizer)
                        self.examples.extend(examples)
                        article_lines = []
                    else:
                        if article_open:
                            article_lines.append(line)

        logger.info("Dataset parse finished.")

    def create_examples_from_document(self, document, block_size, tokenizer, short_seq_prob=0.1):
        """Creates examples for a single document."""

        # Account for special tokens
        max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        examples = []
        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]  # get a segment
            if not segment:
                i += 1
                continue
            current_chunk.append(segment)  # add a segment to current chunk
            current_length += len(segment)  # overall token length
            # if current length goes to the target length or reaches the end of file, start building token a and b
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A` (first) sentence.
                    a_end = 1
                    # if current chunk has more than 2 sentences, pick part of it `A` (first) sentence
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    # token a
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    # token b
                    tokens_b = []
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if len(tokens_a) == 0 or len(tokens_b) == 0:
                        continue

                    # switch tokens_a and tokens_b randomly
                    if random.random() < 0.5:
                        is_next = False
                        tokens_a, tokens_b = tokens_b, tokens_a
                    else:
                        is_next = True

                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break
                            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                            assert len(trunc_tokens) >= 1
                            # We want to sometimes truncate from the front and sometimes from the
                            # back to add more randomness and avoid biases.
                            if random.random() < 0.5:
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # add special tokens
                    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "sentence_order_label": torch.tensor(0 if is_next else 1, dtype=torch.long),
                    }
                    examples.append(example)
                current_chunk = []  # clear current chunk
                current_length = 0  # reset current text length
            i += 1  # go to next line
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

# xxxx - class to create aui and vector file
class AUItoVector(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        outfile_path: str,
        block_size: int,
    ):
        # read line by line and add the aui and tokenized input into a dictionary with aui as key
        # aui_vect = {aui: [token ids of the aui string]}
        print(f"Creating aui_vec.pkl file ....")
        
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        aui_vect = {}
        with open(file_path, 'r') as rf:
            for _, l in enumerate(rf):
                aui = l.split("|")[0]
                aui_string = l.split("|")[1]
                aui_string = aui_string.strip("\n")
                aui_tokens = tokenizer.tokenize(aui_string)
                aui_tokens = tokenizer.convert_tokens_to_ids(aui_tokens)
                clipped_aui_tokens = aui_tokens[:self.block_size]
                aui_vect[aui] = clipped_aui_tokens
        print(f"aui_vec dict. created ...")
        print(f"Saving aui_vec dict. to json file ...")
        with open('aui_vec_json.txt', 'w') as jfile:
            jfile.write(json.dumps(aui_vect))
        print(f"Saved ....")
        with open(outfile_path, "wb") as handle:
            pickle.dump(aui_vect, handle, protocol=pickle.HIGHEST_PROTOCOL)


class StreamDatasetforSynonymyPrediction(Dataset):
    """
    Load data line by line 
    
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        aui_vec_file_path: str,
        data_file_path: str,
        # block_size: int,
    ):

        self.tokenizer = tokenizer
        # open aui_vec file and load to RAM
        self.aui_vec = {}
        print(f"Start loading aui_vec pkl")
        with open(aui_vec_file_path, "rb") as handle:
            self.aui_vec = pickle.load(handle)
        

        # load train dataset to RAM
        print(f"Start loading data file")
        self.examples = []
        with open(data_file_path, "r") as rf:
            # self.examples = rf.readlines()
            self.examples = [line.strip() for line in rf]
        
        # print(type(self.examples))
        # print(self.examples[1])
        # print(len(self.examples))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        self.idx = i
        aui_1 = self.examples[self.idx].split("|")[1]
        aui1_vec = self.aui_vec[aui_1]
        aui_2 = self.examples[self.idx].split("|")[2]
        aui2_vec = self.aui_vec[aui_2]
        synonymy_label = self.examples[self.idx].split("|")[3]
        synonymy_label_int = int(synonymy_label)
        # print(f"type synonymy_label_int: {type(synonymy_label_int)}")
        # print(f"aui1: {aui_1}, aui2: {aui_2}, original_label: {synonymy_label_int}")
        
        input_ids = self.tokenizer.build_inputs_with_special_tokens(aui1_vec, aui2_vec)
        token_type_ids  =self.tokenizer.create_token_type_ids_from_sequences(aui1_vec, aui2_vec)
    
        example = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "next_sentence_label": torch.tensor(0 if synonymy_label_int==1 else 1, dtype=torch.long),
        }
        # return self.examples[i]
        return example

class StreamDatasetforSapBERT(Dataset):
    """
    Load data line by line 
    
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        aui_vec_file_path: str,
        data_file_path: str,
        # block_size: int,
    ):

        self.tokenizer = tokenizer
        # open aui_vec file and load to RAM
        self.aui_vec = {}
        print(f"Start loading aui_vec pkl")
        with open(aui_vec_file_path, "rb") as handle:
            self.aui_vec = pickle.load(handle)
        

        # load train dataset to RAM
        print(f"Start loading data file")
        self.examples = []
        with open(data_file_path, "r") as rf:
            # self.examples = rf.readlines()
            self.examples = [line.strip() for line in rf]
        
        # print(type(self.examples))
        # print(self.examples[1])
        # print(len(self.examples))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        self.idx = i
        aui_1 = self.examples[self.idx].split("|")[1]
        aui1_vec = self.aui_vec[aui_1]
        aui_2 = self.examples[self.idx].split("|")[2]
        aui2_vec = self.aui_vec[aui_2]
        synonymy_label = self.examples[self.idx].split("|")[3]
        synonymy_label_int = int(synonymy_label)
        
        input_ids = self.tokenizer.build_inputs_with_special_tokens(aui1_vec, aui2_vec)
        token_type_ids  =self.tokenizer.create_token_type_ids_from_sequences(aui1_vec, aui2_vec)
    
        example = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "next_sentence_label": torch.tensor(1 if synonymy_label_int==1 else 0, dtype=torch.long),
        }
        return example

class TextDatasetforSynonymyPrediction(Dataset):
    """
    Method for creating synonymy prediction dataset
    Following labels are used in nsp task
    if a sentence is next sentence then then the label is 0
    if a sentence is not next sentence then the label is 1
    So basically if nsp is true it is represented by having a a label with value 0 and, if nsp is false it is
    represented by having a label with the value 1.
    So we do something similar:
    if two terms are synonyms or they currently have the label one from original dataset we convert it to
    0 to be compatible with the nsp code. Therefore out final labels would look like below:
        synonymy_label = 0 (terms are synonyms)
        synonymy_label = 1 (terms are not synonyms)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
    ):
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_synonymy_prediction_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer = tokenizer
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                # xxxx - streaming pickle
                self.examples = []
                with open(cached_features_file, "rb") as handle:
                    while True:
                        try:
                            self.examples.append(pickle.load(handle))
                        except EOFError:
                            break
                '''
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                '''
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
                
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                self.documents = []

                with open(file_path, encoding='utf-8') as f:
                    for _, line in enumerate(f):
                        line = line.strip()
                        if line:
                            single_document = []
                            syn1, syn2, synlabel = line.split("|")
                            syn1_tokens = tokenizer.tokenize(syn1)
                            syn1_tokens = tokenizer.convert_tokens_to_ids(syn1_tokens)
                            syn2_tokens = tokenizer.tokenize(syn2)
                            syn2_tokens = tokenizer.convert_tokens_to_ids(syn2_tokens)
                            if all([syn1_tokens, syn1_tokens, synlabel]):
                                # can add for loop
                                single_document.append(syn1_tokens)
                                single_document.append(syn2_tokens)
                                single_document.append(synlabel)
                                self.documents.append(single_document)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index) 
                
                start = time.time()
                # xxxx - streaming pickle
                with open(cached_features_file, "wb") as handle:
                    for l in self.examples:
                        pickle.dump(l, handle)
                '''
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                '''
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int):

        # if token_length is more than block_size; clip token sequence to block size
        syn1 = document[0][:self.block_size]
        syn2 = document[1][:self.block_size]
        synlabel = int(document[2])

        # add special tokens
        input_ids = self.tokenizer.build_inputs_with_special_tokens(syn1, syn2)
        # add token type ids, 0 for sentence a, 1 for sentence b
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(syn1, syn2)

        example = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            # keep as is 0=0, 1=1
            "next_sentence_label": torch.tensor(0 if synlabel==1 else 1, dtype=torch.long),
        }
        self.examples.append(example)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]

class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_nsp_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer = tokenizer

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                # print(self.documents)
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        # print(f"self.documents: {self.documents}")
                        line = f.readline()
                        # print(type(line))
                        if not line:
                            break
                        line = line.strip()
                        # print(f"stripped line: {line}")

                        # Empty lines are used as document delimiters
                        # print(f"self.documents: {self.documents}")
                        # print(f"self.documents[-1]: {self.documents[-1]}")
                        # print(f"len(self.documents[-1]): {len(self.documents[-1])}")
                        # print(f"if not line: {not line}")
                        # print(f"len(self.documents[-1]) != 0: {len(self.documents[-1]) != 0}")
                        if not line and len(self.documents[-1]) != 0:
                            # print(f"not end of file and prev line not empty: {self.documents}")
                            self.documents.append([])
                            # print(f"after this append self.documents.append([]): {self.documents}")
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            # print(f"appending tokens to this location self.documents[-1]: {self.documents[-1]}")
                            self.documents[-1].append(tokens)
                            # print(f"self.documents[-1]--- after appending tokens: {self.documents}")
                # print(f"how --self.documents-- look like: {self.documents}")
                # print(f"Creating examples from {len(self.documents)} documents.")
                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int):
        """Creates examples for a single document."""
        print("\n\n\n\n\n")
        print(f"START OF NEW DOC!!")
        print(document, doc_index)
        # print(f"block size: {self.block_size}")
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)
        # print(f"max_num_tokens: {max_num_tokens}")

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        
        target_seq_length = max_num_tokens
        # print(f"target_seq_length: {target_seq_length}")
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0
        print(f"document: {document}")
        print(f"len(document): {len(document)}")
        while i < len(document):
            print(f"i: {i}")
            segment = document[i]
            print(f"segment: {segment}")
            current_chunk.append(segment)
            current_length += len(segment)
            print(f"current_length: {current_length}")
            if i == len(document) - 1 or current_length >= target_seq_length:
                print(f"i == len(document) - 1 or current_length >= target_seq_length {i == len(document) - 1} {current_length >= target_seq_length}")
                if current_chunk:
                    print(f"current_chunk = {current_chunk}")
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    print(f"len(current_chunk) >= 2: {len(current_chunk)}")
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                        print(f"a_end: {a_end}")

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    print(f"len(current_chunk): {len(current_chunk)}")
                    if len(current_chunk) == 1 or random.random() < self.nsp_probability:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)
                        print(f"target_b_length: {target_b_length}")

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(0, len(self.documents) - 1)
                            if random_document_index != doc_index:
                                break

                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        print(f"a_end: {a_end}")
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    print(f"tokens_a: {tokens_a} \ntokens_b: {tokens_b}")
                    # add special tokens
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.long),
                    }
                    print(example)

                    self.examples.append(example)
                    
                current_chunk = []
                current_length = 0

            i += 1
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
