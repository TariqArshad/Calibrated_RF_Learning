import torch

class Conv1D_RF_Classifier(torch.nn.Module):
    seq_len = 1024
    def __init__(self, classes = 3):
        super(Conv1D_RF_Classifier, self).__init__()
        """
        1D CNN that takes in sequences of 1024 with 4 channles consisting of [real_component, imag_component, Magnitude, Phase]
        #TO-DO:make seq_len a changeable variable 
        """
        self.loss_fn = torch.nn.CrossEntropyLoss();
        self.classes = classes;
        
        self.conv_0 = torch.nn.Conv1d(in_channels = 4, out_channels = 64, kernel_size = 9, stride = 1);
        self.batch_norm_0 = torch.nn.BatchNorm1d(64);
        self.max_pool_0 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.conv_1 = torch.nn.Conv1d(in_channels = 64, out_channels = 32, kernel_size = 9, stride = 1);
        self.batch_norm_1 = torch.nn.BatchNorm1d(32)
        self.max_pool_1 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.conv_2 = torch.nn.Conv1d(in_channels = 32 , out_channels = 16, kernel_size = 9, stride = 1);  
        self.batch_norm_2 = torch.nn.BatchNorm1d(16);
        self.max_pool_2 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.linear_0 = torch.nn.Linear(1936, 256); #for 1024 seq len
        self.linear_1 = torch.nn.Linear(256,classes);
        self.Relu = torch.nn.ReLU()
        self.Conv_Dropout = torch.nn.Dropout(0.5);
        self.Linear_Dropout = torch.nn.Dropout(0.3);
         

    def forward(self, data_in, y_true = None):
        """
        data_in shape:(batch_sz,channel_sz,seq_len), where channel_sz = 4, seq_ken = 1024
        """

        logits = self.conv_0(data_in)
        logits = self.Relu(logits)
        logits = self.batch_norm_0(logits)
        logits = self.max_pool_0(logits)
        logits = self.Conv_Dropout(logits)
        
        logits = self.conv_1(logits)
        logits = self.Relu(logits)
        logits = self.batch_norm_1(logits)
        logits = self.max_pool_1(logits)
        logits = self.Conv_Dropout(logits)
        
        logits = self.conv_2(logits)
        logits = self.Relu(logits)
        logits = self.batch_norm_2(logits)
        logits = self.max_pool_2(logits)
        logits = self.Conv_Dropout(logits)
    
        logits = logits.flatten(1, 2)
        logits = self.linear_0(logits)
        logits = self.Relu(logits)
        logits = self.Linear_Dropout(logits)

        logits = self.linear_1(logits)
        
        if self.training:
            logits = logits.reshape((-1, logits.shape[1]))
            probs = torch.nn.functional.softmax(logits, dim = -1)
            loss = self.loss_fn(logits, y_true)
            return logits, loss
        else:
            probs = torch.nn.functional.softmax(logits, dim = -1)
            preds = torch.argmax(probs, dim = -1)
            preds = torch.nn.functional.one_hot(preds, num_classes = probs.shape[-1])
            return logits, probs ,preds #preds is binary vector with 1's on all classes who prob is more than threshold



class Conv1D_RF_MultiClassifier(torch.nn.Module):
    seq_len = 1024
    conf_thresh = 0.50
    def __init__(self, classes = 3, channels = 4):
        super(Conv1D_RF_MultiClassifier, self).__init__()
        """
        Multilabel 1D CNN that takes in sequences of 1024 with 4 channles consisting of [real_component, imag_component, Magnitude, Phase]
        with binary classifier for each label 
        """
        self.loss_fn = torch.nn.BCELoss()
        self.classes = classes

        self.conv_0 = torch.nn.Conv1d(in_channels = channels, out_channels = 256, kernel_size = 9, stride = 1)
        self.batch_norm_0 = torch.nn.BatchNorm1d(256)
        self.max_pool_0 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)

        self.conv_1 = torch.nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size = 9, stride = 1)
        self.batch_norm_1 = torch.nn.BatchNorm1d(256)
        self.max_pool_1 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.conv_2 = torch.nn.Conv1d(in_channels = 256, out_channels = 128, kernel_size = 9, stride = 1)
        self.batch_norm_2 = torch.nn.BatchNorm1d(128)
        self.max_pool_2 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.conv_3 = torch.nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 9, stride = 1)
        self.batch_norm_3 = torch.nn.BatchNorm1d(64)
        self.max_pool_3 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.conv_4 = torch.nn.Conv1d(in_channels = 64 , out_channels = 32, kernel_size = 9, stride = 1)  
        self.batch_norm_4 = torch.nn.BatchNorm1d(32)
        self.max_pool_4 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.linear_0 = torch.nn.Linear(768, 256)
        self.linear_1 = torch.nn.Linear(256,classes)
        self.Relu = torch.nn.ReLU()
        self.Conv_Dropout = torch.nn.Dropout(0.5)
        self.Linear_Dropout = torch.nn.Dropout(0.3)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data_in, y_true = None):
        """
        - data_in shape:(batch_sz,channel_sz,seq_len), where channel_sz = 4, seq_ken = 1024
        - conf_thresh: confidence threshold for binary classifier
        - y_true: ground truth in binary encoding format
        """

        logits = self.conv_0(data_in)
        logits = self.Relu(logits)
        logits = self.batch_norm_0(logits)
        logits = self.max_pool_0(logits)
        logits = self.Conv_Dropout(logits)

        logits = self.conv_1(logits)
        logits = self.Relu(logits)
        logits = self.batch_norm_1(logits)
        logits = self.max_pool_1(logits)
        logits = self.Conv_Dropout(logits)

        logits = self.conv_2(logits)
        logits = self.Relu(logits)
        logits = self.batch_norm_2(logits)
        logits = self.max_pool_2(logits)
        logits = self.Conv_Dropout(logits)
    
        logits = self.conv_3(logits)
        logits = self.Relu(logits)
        logits = self.batch_norm_3(logits)
        logits = self.max_pool_3(logits)
        logits = self.Conv_Dropout(logits)

        logits = self.conv_4(logits)
        logits = self.Relu(logits)
        logits = self.batch_norm_4(logits)
        logits = self.max_pool_4(logits)
        logits = self.Conv_Dropout(logits)
        
        logits = logits.flatten(1, 2)
        logits = self.linear_0(logits)
        logits = self.Relu(logits)
        logits = self.Linear_Dropout(logits)

        logits = self.linear_1(logits)

      
        if self.training:
            logits = logits.reshape((-1, logits.shape[1]))
            probs = self.sigmoid(logits)
            loss = self.loss_fn(probs, y_true)
            return logits, loss
        else:
            probs = self.sigmoid(logits)
            multi_preds = probs > self.conf_thresh
            return logits, probs, multi_preds #preds is binary vector with 1's on all classes who prob is more than threshold
