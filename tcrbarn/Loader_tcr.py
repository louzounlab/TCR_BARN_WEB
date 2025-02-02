from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import tcrbarn.v_j_tcr as v_j_tcr


class ChainClassificationDataset(Dataset):
    """
    Dataset class for chain pairing classification.
    Args:
        data (list): List of data samples.
        vj_data (tuple): Tuple containing dictionaries for V and J gene encodings.
    """
    def __init__(self, data, vj_data):
        self.data = data
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                       'Y']
        # Special tokens
        PAD_TOKEN = "<PAD>"
        EOS_TOKEN = "<EOS>"
        SOS_TOKEN = "<SOS>"
        UNK_TOKEN = "<UNK>"
        self.vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(amino_acids)
        self.acid_2_ix = {word: i for i, word in enumerate(self.vocab)}
        self.ix_2_acid = {i: word for word, i in self.acid_2_ix.items()}
        # Initialize one-hot encodings for V and J genes
        va_dict, vb_dict, ja_dict, jb_dict = vj_data
        self.va_2_ix = va_dict
        self.vb_2_ix = vb_dict
        self.ja_2_ix = ja_dict
        self.jb_2_ix = jb_dict

        self.num_va = len(va_dict)
        self.num_vb = len(vb_dict)
        self.num_ja = len(ja_dict)
        self.num_jb = len(jb_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        """
        Get a sample from the dataset.
        Args:
            ix (int): Index of the sample.
        Returns:
            tuple: Tensors for alpha chain, beta chain, V and J gene one-hot encodings, label, and output.
        """
        if len(self.data[ix]) == 2:
            chain_pair, vj = self.data[ix]
            output = None
            label = None
        elif len(self.data[ix]) == 3:
            chain_pair, vj, label = self.data[ix]
            output = None
        elif len(self.data[ix]) == 4:
            chain_pair, vj, label, output = self.data[ix]
            if label == -1:
                label = None
            output = float(output)  # Ensure `output` is a float
        else:
            raise ValueError("Unexpected number of columns in dataset row.")
        chain1, chain2 = chain_pair
        if label is not None:
            label = float(label)
        # Convert amino acid sequences to tensors
        alpha_tensor = torch.tensor([self.acid_2_ix[acid] for acid in chain1] + [self.acid_2_ix["<EOS>"]],
                                    dtype=torch.long)
        beta_tensor = torch.tensor([self.acid_2_ix[acid] for acid in chain2] + [self.acid_2_ix["<EOS>"]],
                                   dtype=torch.long)
        # One-hot encode V and J genes
        va, vb, ja, jb = vj
        va_one_hot = torch.zeros(self.num_va)
        va_one_hot[self.va_2_ix.get(va, self.va_2_ix["<UNK>"])] = 1
        vb_one_hot = torch.zeros(self.num_vb)
        vb_one_hot[self.vb_2_ix.get(vb, self.vb_2_ix["<UNK>"])] = 1
        ja_one_hot = torch.zeros(self.num_ja)
        ja_one_hot[self.ja_2_ix.get(ja, self.ja_2_ix["<UNK>"])] = 1
        jb_one_hot = torch.zeros(self.num_jb)
        jb_one_hot[self.jb_2_ix.get(jb, self.jb_2_ix["<UNK>"])] = 1

        return alpha_tensor, beta_tensor, va_one_hot, vb_one_hot, ja_one_hot, jb_one_hot, label, output


def collate_fn(batch):
    """
    Collate function to combine samples into a batch.
    Args:
        batch (list): List of samples.
    Returns:
        tuple: Batched tensors for input sequences, V and J gene encodings, labels, and outputs.
    """
    chain1_batch, chain2_batch, va_one_hot, vb_one_hot, ja_one_hot, jb_one_hot, label_batch, output = zip(*batch)
    # Pad input chains
    input_tensor1_padded = pad_sequence(chain1_batch, batch_first=True, padding_value=0)
    input_tensor2_padded = pad_sequence(chain2_batch, batch_first=True, padding_value=0)
    # Stack one-hot encoded tensors into a batch
    va_one_hot_tensor = torch.stack(va_one_hot)
    vb_one_hot_tensor = torch.stack(vb_one_hot)
    ja_one_hot_tensor = torch.stack(ja_one_hot)
    jb_one_hot_tensor = torch.stack(jb_one_hot)
    if label_batch is not None and any(o is not None for o in label_batch):
        label_batch = torch.tensor(label_batch, dtype=torch.float)
    if output is not None and any(o is not None for o in output):
        output = torch.tensor(output, dtype=torch.float)
    return (input_tensor1_padded, input_tensor2_padded, va_one_hot_tensor, vb_one_hot_tensor, ja_one_hot_tensor,
            jb_one_hot_tensor, label_batch, output)


def read_data(file_path, chain1_column='tcra', chain2_column='tcrb', va_c='va', vb_c="vb", ja_c="ja", jb_c="jb",
              label_column='sign'):
    """
    Read data from a CSV file and format it for the dataset.
    Args:
        file_path (str): Path to the CSV file.
        chain1_column (str, optional): Column name for the first chain. Default is 'tcra'.
        chain2_column (str, optional): Column name for the second chain. Default is 'tcrb'.
        va_c (str, optional): Column name for the V gene of the first chain. Default is 'va'.
        vb_c (str, optional): Column name for the V gene of the second chain. Default is 'vb'.
        ja_c (str, optional): Column name for the J gene of the first chain. Default is 'ja'.
        jb_c (str, optional): Column name for the J gene of the second chain. Default is 'jb'.
        label_column (str, optional): Column name for the label. Default is 'sign'.
    Returns:
        list: List of formatted data samples.
    """
    df = pd.read_csv(file_path)
    if df.shape[1] == 7:
        output_col = False  # Default if `output` column is not present
    else:
        output_col = True
    # Extract chains and label from each row
    data = []
    for _, row in df.iterrows():
        chain1 = row[chain1_column]
        chain2 = row[chain2_column]
        label = row[label_column]
        va = v_j_tcr.v_j_format(row[va_c], 1, "TRAV")
        vb = v_j_tcr.v_j_format(row[vb_c], 1, "TRBV")
        ja = v_j_tcr.v_j_format(row[ja_c], 1, "TRAJ")
        jb = v_j_tcr.v_j_format(row[jb_c], 2, "TRBJ")
        if output_col:
            output_score = row['output']
            data.append([(chain1, chain2), (va, vb, ja, jb), label, output_score])
        else:
            data.append([(chain1, chain2), (va, vb, ja, jb), label])
    return data
