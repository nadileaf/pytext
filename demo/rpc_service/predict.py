import jieba
import numpy as np
import sys
sys.path.append('../../')
from caffe2.python import workspace
from caffe2.python.predictor import predictor_exporter
from demo.rpc_service.config import MODEL_FILE, MODEL_DB_TYPE


class Entity(object):
    def __init__(self,
                 value: str,
                 start: int,
                 end: int,
                 entity: str):
        self.value = value
        self.start = start
        self.end = end
        self.entity = entity

    def serialize(self):
        return {
            "value": self.value,
            "start": self.start,
            "end": self.end,
            "entity": self.entity
        }


class Predictor(object):
    def __init__(self):
        self.predict_net = predictor_exporter.prepare_prediction_net(
                            filename=MODEL_FILE, db_type=MODEL_DB_TYPE)

    # Pre-processing helper method
    def featurize(self, text):
        tokens = []
        token_ranges = []

        result = jieba.tokenize(text)
        for tk in result:
            tokens.append(tk[0])
            token_ranges.append((tk[1], tk[2]))

        if not tokens:
            # Add PAD_TOKEN in case of empty text
            tokens = ["<pad>"]
        tokens = list(map(str.lower, tokens))
        return tokens, token_ranges

    # Run ATIS model
    def predict(self, text):
        # Pre-process
        tokens, token_ranges = self.featurize(text)

        # Make prediction
        workspace.blobs["tokens_vals_str:value"] = np.array([tokens], dtype=str)
        workspace.blobs["tokens_lens"] = np.array([len(tokens)], dtype=np.int_)
        workspace.RunNet(self.predict_net)
        labels_scores = [
            (str(blob), workspace.blobs[blob][0])
            for blob in self.predict_net.external_outputs
            if "word_scores" in str(blob)
        ]
        labels = list(zip(*labels_scores))[0]
        scores = list(zip(*labels_scores))[1]  # len(tokens) x 1

        # Post-processing (find city names)
        all_scores = np.concatenate(scores, axis=1)  # len(tokens) x len(labels)
        predicted_labels = np.argmax(all_scores, axis=1)  # len(tokens)

        # city_token_ranges = []
        res = []
        prev_token_range = (0, 0)
        prev_value = ""
        prev_label = 'O'
        for token_idx, label_idx in enumerate(predicted_labels):
            if 'NoLabel' in labels[label_idx]:
                label = 'O'
            else:
                label = labels[label_idx].split('-')[-1]
            if prev_label == label or token_idx == 0:
                prev_value += tokens[token_idx]
                prev_label = label
                prev_token_range = (prev_token_range[0], token_ranges[token_idx][1])
            else:
                res.append(Entity(value=prev_value, start=prev_token_range[0],
                                  end=prev_token_range[1], entity=prev_label).serialize())
                prev_label = label
                prev_token_range = token_ranges[token_idx]
                prev_value = tokens[token_idx]
        res.append(Entity(value=prev_value, start=prev_token_range[0],
                          end=prev_token_range[1], entity=prev_label).serialize())
        return res


if __name__ == '__main__':
    text = '今日头条 与阿里巴巴达成战略合作'
    p = Predictor()
    res = p.predict(text)
    print(res)