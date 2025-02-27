import torch
import torch.nn as nn

class Trainer:
    ''' Class to train classificator or reconstructor '''
    def __init__(self,
                 task,
                 load_embeddings,
                 net,
                 class_weights,
                 optim, gradient_clipping_value,
                 scaler=None):
        # Store vars
        self.task = task
        self.load_embeddings = load_embeddings
        # Store model
        self.net = net
        # Store optimizer
        self.optim = optim
        # Create Loss
        self.criterion = None
        if self.task == 'classification':
            self.criterion = nn.CrossEntropyLoss(weight = class_weights)
        elif self.task == 'reconstruction':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Task not recognized')
        # Gradient clipping
        self.gradient_clipping_value = gradient_clipping_value
        # CUDA AMP
        self.scaler = scaler

    def forward_batch(self, input_datas, labels, split):
        ''' send a batch to net and backpropagate '''
        def forward_batch_part():
            # Set network mode
            if split == 'train':
                self.net.train()
                torch.set_grad_enabled(True)   
            else:
                self.net.eval()
                torch.set_grad_enabled(False)
            
            if self.scaler is None:
                if input_datas is None:
                    raise ValueError('Provided input is None')
                elif isinstance(input_datas, list) and len(input_datas) == 0:
                    raise ValueError('All inputs are disabled')
                
                if self.task == 'classification':
                    if self.load_embeddings:
                        x = input_datas
                    else:
                        # Preprocess input
                        if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
                            x = self.net.module.preprocess_input(input_datas)
                        else:
                            x = self.net.preprocess_input(input_datas)
                    
                    # Foward pass
                    predicted_labels_logits = self.net(x)
                    
                    # Compute loss
                    loss = self.criterion(predicted_labels_logits, labels)
                elif self.task == 'reconstruction':
                    # Preprocess input
                    if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
                        x, (len_x, len_batch, len_ch_time, len_time) = self.net.module.preprocess_input(input_datas)
                    else:
                        x, (len_x, len_batch, len_ch_time, len_time) = self.net.preprocess_input(input_datas)
                    
                    # Foward pass
                    x_reconstruct = self.net(x)[0]
                    
                    # Compute loss
                    loss = self.criterion(x_reconstruct, x)
                
                    # Postprocess output
                    #if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
                    #    output_datas = self.net.module.postprocess_output(x_reconstruct, *(len_x, len_batch, len_ch_time, len_time))
                    #else:
                    #    output_datas = self.net.postprocess_output(x_reconstruct, *(len_x, len_batch, len_ch_time, len_time))
                else:
                    raise ValueError('Task not recognized')
            else:
                with torch.amp.autocast(device_type=input_datas[0].device.type if isinstance(input_datas, list) else input_datas.device.type):
                    if input_datas is None:
                        raise ValueError('Provided input is None')
                    elif isinstance(input_datas, list) and len(input_datas) == 0:
                        raise ValueError('All inputs are disabled')
                    
                    if self.task == 'classification':
                        if self.load_embeddings:
                            x = input_datas
                        else:
                            # Preprocess input
                            if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
                                x = self.net.module.preprocess_input(input_datas)
                            else:
                                x = self.net.preprocess_input(input_datas)
                        
                        # Foward pass
                        predicted_labels_logits = self.net(x)
                        
                        # Compute loss
                        loss = self.criterion(predicted_labels_logits, labels)
                    elif self.task == 'reconstruction':
                        # Preprocess input
                        if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
                            x, (len_x, len_batch, len_ch_time, len_time) = self.net.module.preprocess_input(input_datas)
                        else:
                            x, (len_x, len_batch, len_ch_time, len_time) = self.net.preprocess_input(input_datas)
                        
                        # Foward pass
                        x_reconstruct = self.net(x)[0]
                        
                        # Compute loss
                        loss = self.criterion(x_reconstruct, x)
                    
                        # Postprocess output
                        #if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
                        #    output_datas = self.net.module.postprocess_output(x_reconstruct, *(len_x, len_batch, len_ch_time, len_time))
                        #else:
                        #    output_datas = self.net.postprocess_output(x_reconstruct, *(len_x, len_batch, len_ch_time, len_time))
            
            if self.task == 'classification':
                # Calculate label predicted and scores
                _, predicted_labels = torch.max(predicted_labels_logits.data, 1)
                predicted_scores = predicted_labels_logits.data.clone().detach().cpu()
            elif self.task == 'reconstruction':
                predicted_labels = None
                predicted_scores = None
            else:
                raise ValueError('Task not recognized')

            if split == 'train':
                # Zero the gradient
                self.optim.zero_grad()

                # Backpropagate
                if self.scaler is None:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clipping_value > 0:
                    if self.scaler is None:
                        torch.nn.utils.clip_grad_value_(self.net.parameters(), self.gradient_clipping_value)
                    else:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_value_(self.net.parameters(), self.gradient_clipping_value)
            
            return loss, (predicted_labels, predicted_scores)

        loss, (predicted_labels, predicted_scores) = forward_batch_part()

        if not isinstance(self.optim,torch.optim.LBFGS):
            if split == 'train':
                # Update weights (and scaler if exists)
                if self.scaler is None:
                    self.optim.step()
                else:
                    self.scaler.step(self.optim)
                    self.scaler.update()
        else:
            if split == 'train':
                # Update weights (and scaler if exists)
                if self.scaler is None:
                    self.optim.step(forward_batch_part)
                else:
                    self.scaler.step(self.optim, forward_batch_part)
                    self.scaler.update()
        
        # Metrics
        metrics = {}
        metrics['loss'] = loss.item()

        return metrics, (predicted_labels, predicted_scores)

    def forward_batch_testing(task, load_embeddings, net, input_datas, labels, class_weights, scaler=None):
        ''' Send a batch to net and backpropagate '''
        # Create Loss
        criterion = None
        if task == 'classification':
            criterion = nn.CrossEntropyLoss(weight = class_weights)
        elif task == 'reconstruction':
            criterion = nn.MSELoss()
        else:
            raise ValueError('Task not recognized')
        
        # Set network mode
        net.eval()
        torch.set_grad_enabled(False)
        
        if scaler is None:
            if input_datas is None:
                raise ValueError('Provided input is None')
            elif isinstance(input_datas, list) and len(input_datas) == 0:
                raise ValueError('All inputs are disabled')
                
            if task == 'classification':
                if load_embeddings:
                    x = input_datas
                else:
                    # Preprocess input
                    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        x = net.module.preprocess_input(input_datas)
                    else:
                        x = net.preprocess_input(input_datas)
                
                # Foward pass
                predicted_labels_logits = net(x)
                
                # Compute loss
                loss = criterion(predicted_labels_logits, labels)
            elif task == 'reconstruction':
                # Preprocess input
                if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    x, (len_x, len_batch, len_ch_time, len_time) = net.module.preprocess_input(input_datas)
                else:
                    x, (len_x, len_batch, len_ch_time, len_time) = net.preprocess_input(input_datas)
                
                # Foward pass
                x_reconstruct = net(x)[0]
                
                # Compute loss
                loss = criterion(x_reconstruct, x)
            
                # Postprocess output
                #if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                #    output_datas = net.module.postprocess_output(x_reconstruct, *(len_x, len_batch, len_ch_time, len_time))
                #else:
                #    output_datas = net.postprocess_output(x_reconstruct, *(len_x, len_batch, len_ch_time, len_time))
        else:
            with torch.amp.autocast(device_type=input_datas[0].device.type if isinstance(input_datas, list) else input_datas.device.type):
                if input_datas is None:
                    raise ValueError('Provided input is None')
                elif isinstance(input_datas, list) and len(input_datas) == 0:
                    raise ValueError('All inputs are disabled')
                
                if task == 'classification':
                    if load_embeddings:
                        x = input_datas
                    else:
                        # Preprocess input
                        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                            x = net.module.preprocess_input(input_datas)
                        else:
                            x = net.preprocess_input(input_datas)
                    
                    # Foward pass
                    predicted_labels_logits = net(x)
                    
                    # Compute loss
                    loss = criterion(predicted_labels_logits, labels)
                elif task == 'reconstruction':
                    # Preprocess input
                    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        x, (len_x, len_batch, len_ch_time, len_time) = net.module.preprocess_input(input_datas)
                    else:
                        x, (len_x, len_batch, len_ch_time, len_time) = net.preprocess_input(input_datas)
                    
                    # Foward pass
                    x_reconstruct = net(x)[0]
                    
                    # Compute loss
                    loss = criterion(x_reconstruct, x)
                
                    # Postprocess output
                    #if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    #    output_datas = net.module.postprocess_output(x_reconstruct, *(len_x, len_batch, len_ch_time, len_time))
                    #else:
                    #    output_datas = net.postprocess_output(x_reconstruct, *(len_x, len_batch, len_ch_time, len_time))
        
        if task == 'classification':
            # Calculate label predicted and scores
            _, predicted_labels = torch.max(predicted_labels_logits.data, 1)
            predicted_scores = predicted_labels_logits.data.clone().detach().cpu()
        elif task == 'reconstruction':
            predicted_labels = None
            predicted_scores = None
        else:
            raise ValueError('Task not recognized')
        
        # Metrics
        metrics = {}
        metrics['loss'] = loss.item()

        return metrics, (predicted_labels, predicted_scores)