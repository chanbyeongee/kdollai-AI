from transformers import TFBertModel
import tensorflow as tf

#다대일, 감정분석
class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name, num_labels):
        super(TFBertForSequenceClassification, self).__init__()
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