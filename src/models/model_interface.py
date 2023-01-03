# interface class, each model class need to implement these functions
class ModelInterface():
    def predict(self, data_frame, model_dir, labels):
        raise NotImplementedError("Interface class, override 'predict()' function in child class")

    def train(self, train_frame, labels, test_frame=None):
        raise NotImplementedError("Interface class, override 'train()' function in child class")

    def evaluate(self):
        raise NotImplementedError("Interface class, override 'evaluate()' function in child class")
