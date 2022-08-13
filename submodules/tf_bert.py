from transformers import TFBertModel
import tensorflow as tf

#다대일, 감정분석
class SequenceClassification(tf.keras.Model):
    def __init__(self, model_name, num_labels):
        super(SequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.drop = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_labels,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range),
                                                activation='softmax',
                                                name='classifier')

    def call(self, inputs, training=None, mask=None):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        #Bert의 값이 encoder (pooler)값이 반환됨 64*29
        output = outputs[1]
        dropped = self.drop(output, training=False)
        prediction = self.classifier(dropped)

        return prediction


#다대다 개체명인식
class TokenClassification(tf.keras.Model):
    def __init__(self, model_name, labels):
        super(TokenClassification, self).__init__()
        # 모델 구조 생성 (64 x 128 x 29)
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.drop = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(labels,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                name='classifier')

    def call(self, inputs, training=None, mask=None):
        # encoding input, mask, positional encoding
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Bert의 값이 (encoder)/ pooler값이 반환됨 64*128*29 (단어마다 매김)
        all_output = outputs[0]
        prediction = self.classifier(all_output)

        return prediction