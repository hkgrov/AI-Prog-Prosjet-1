import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import mnist_basics as mnists
import tflowtools as TFT
import json
import time
from pprint import pprint
import random
from io import BytesIO
from collections import defaultdict
import pandas as pd
import seaborn as sns

__mnist_path__ = "/Users/hakongrov/Documents/INDØK/4.År/AI-Prog/Files/dataset/mnist-zip/"


# ******* A General Artificial Neural Network ********
# This is the original GANN, which has been improved in the file gann.py

#Global variable including all the functions for generating different datasets. The name of the dataset is the key in the dict.
_generator = {
	"symmetry": lambda length,count : TFT.gen_symvect_dataset(length,count), #Needs spesific length and number of cases: length, count
    "parity": lambda a: TFT.gen_all_parity_cases(a), #Need spesific length parameter, Double flag?
	"autoencoder": lambda a: (TFT.gen_all_one_hot_cases(a)), #or TFT.gen_dense_autoencoder_cases(count, size) #Needs a spesific length and size for the last a spesific density of one's
	"bit counter": lambda num,size: TFT.gen_vector_count_cases(num,size), #Dimensions = [same as input, hidden, input + 1]
	"segment counter":lambda a,b,c,d: TFT.gen_segmented_vector_cases(a, b, c, d)
}


#Global variable for the hidden activation functions.
_hidden_activation_function = {
	"sigmoid": lambda a, name: tf.nn.sigmoid(a, name),
	"relu": lambda a, name: tf.nn.relu(a,name),
	"tanh": lambda a, name: tf.nn.tanh(a, name)
}


#Global variable for the optimizer functions. 
_optimizer = {
    "gradientDescent":lambda lrate:  tf.train.GradientDescentOptimizer(lrate),
    "adagradOptimizer":lambda a: tf.train.AdagradOptimizer(a),
    "adamOptimizer":lambda a: tf.train.AdamOptimizer(a),
    "RMSPropOptimizer":lambda a: tf.train.RMSPropOptimizer(a)
}

#Global variable for the softmax function, but could also add other output activation functions here.
_output_activation_function = {
    "softmax":lambda a:  tf.nn.softmax(a)
}

#Global variable for the cost function used when calculating the error of the output.
_loss_function = {
    "mse": lambda labels, predictions: tf.losses.mean_squared_error(labels, predictions),
    "cross_entropy": lambda labels, targets: tf.losses.softmax_cross_entropy(labels, targets)
}

class Gann():

    def __init__(self, dims, cman,lrate=.1,showint=None,mbs=10,vint=None,softmax=False, config=None):
        #Arrays used to store the name of the modules I want to graph. bname = bias, Hname = hinton plot, Dnames = dendrogram, wnames = weights
        self.bnames = []
        self.Hnames = []
        self.Dnames = []
        self.wnames = []
        #Other variables needed to configure the neural network. 
        self.learning_rate = lrate
        self.layer_sizes = dims # Sizes of each layer of neurons
        self.show_interval = showint # Frequency of showing grabbed variables
        self.global_training_step = 0 # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = [] # One matplotlib figure for each grabvar
        #self.grabvar_figures = defaultdict(list)
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = cman
        self.softmax_outputs = softmax
        self.modules = []
        self.config = config
        self.build()
        #Functions to grab variables to graph and clean them so I don't run them multiple times.
        self.grab_vars(self.config)
        self.clean_vars()
        
        
        



    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input; insize = num_inputs
        # Build all of the modules
        for i,outsize in enumerate(self.layer_sizes[1:]):
            gmod = Gannmodule(self,i,invar,insize,outsize, self.config)
            invar = gmod.output; insize = gmod.outsize
        self.output = gmod.output # Output of last module is output of whole network
        
        #Checking to see if softmax is set to be used on the output layer of the network.
        if self.softmax_outputs: self.output = _output_activation_function[self.config["output activation function"]](self.output)
        self.target = tf.placeholder(tf.float64,shape=(None,gmod.outsize),name='Target')
        self.configure_learning()
        
    #Function to clean the list of grabbed variables to plot, making it a unique list. 
    def clean_vars(self):
        self.grabvars = list(set(self.grabvars))
        
        
        
    def grab_vars(self, config):
        #Grabbing variables to plot the hinton plots. This is based on the layers I want to visualize decided in the config file.
        for key in config["Map Layers"]:
            for i in config["Map Layers"][key]:
                self.add_grabvar(i, key)
                if(key == "in" and i == 0):
                    self.Hnames.append("Input:0")
                elif(key == "in" and i > 0):
                    self.Hnames.append("Module-"+str(i-1)+"-out:0")
                else:
                    self.Hnames.append("Module-"+str(i)+"-"+str(key)+":0")

        #Same as for hinton plot only here I decide which layer to plot in dendrogram
        for key in config["Map Dendograms"]:
            for i in config["Map Dendograms"][key]:
                self.add_grabvar(i, key)
                if(key == "in" and i == 0):
                    self.Dnames.append("Input:0")
                elif(key == "in" and i > 0):
                    self.Dnames.append("Module-"+str(i-1)+"-out:0")
                else:
                    self.Dnames.append("Module-"+str(i)+"-"+str(key)+":0")
        
        #grabbing variables for weight and bias layers to plot.
        if(config["Display Weights"]):
            for i in config["Display Weights"]:
                self.add_grabvar(i, 'wgt')
                self.wnames.append("Module-"+(str(i)+"-wgt:0"))
        
        if(config["Display Biases"]):
            for i in config["Display Biases"]:
                self.add_grabvar(i, 'bias')
                self.bnames.append("Module-"+(str(i)+"-bias:0"))
        
        
    
        
    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type,spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self,module_index,type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(PLT.figure())
        
        
    
    #def add_grabvar(self, module_index, type='wgt'):
     #   self.grabvars[type].append(self.modules[module_index].getvar(type))
      #  self.grab
    

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    #Adding another module (layer) to the neural network.
    def add_module(self,module): self.modules.append(module)

    
    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):
        #self.error = tf.reduce_mean(tf.square(self.target - self.output),name='MSE')
        #Here I set the error function to be used in the nerual network when back propagating.
        self.error = _loss_function[self.config["cost function"]](self.target, self.output)

        
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        
        #Setting the optimizer for how to minimize the error
        optimizer = _optimizer[self.config["optimizer"]](self.learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error,name='Backprop')

    """def do_training(self,sess,cases,epochs=100,continued=False, steps=1):
        if not(continued): self.error_history = []
        
        #Epochs are how many times we run through the training set. 
        for i in range(epochs):
            
            #Steps is how many times we run through a new minibatch. If we run three steps with a minibatch of 10 we run through a total of 30 cases.
            error = 0; step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            
            
            #This is the minibatch size and it should be able to set in the config file
            mbs = self.minibatch_size; ncases = len(cases); nmb = math.ceil(ncases/mbs)
            for cstart in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases,cstart+mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _,grabvals,_ = self.run_one_step([self.trainer],gvars,self.probes,session=sess,
                                         feed_dict=feeder,step=step,show_interval=self.show_interval)
                error += grabvals[0]
            self.error_history.append((step, error/nmb))
            self.consider_validation_testing(step,sess)
        self.global_training_step += epochs
        TFT.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
                                  title="",fig=not(continued))"""
                    
    #Method for plotting the hinton plots, dendrogram and display matrics.                
    def do_mapping(self, sess, cases):
        #Separate the cases into inputs and targets
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        
        #Getting the value for the label
        tar = []
        for t in targets:
            tar.append(list(t).index(1))
        
        #Running the network without learning on the map batch
        feeder = {self.input: inputs, self.target: targets}
        testres, grabvals, _ = self.run_one_step(self.predictor, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
        
        
        #Putting the names into an array for easier look up
        names = [x.name for x in self.grabvars]
        zips = zip(names, grabvals)
        #Variables to show which 
        num = 0
        num2 = 0
        #Plotting the different plots
        for i in zips:
            if(i[0] in self.Hnames):
                TFT.hinton_plot(i[1], fig=PLT.figure(), title= i[0]+ ' at step '+ str("Test"))
            if(i[0] in self.Dnames):
                TFT.dendrogram(i[1], tar, title="dendrogram"+ str(i[0]))
            if(i[0] in self.wnames):
                fig_wgt = PLT.figure()
                TFT.display_matrix(i[1],fig=fig_wgt, title=(i[0]))
                num += 1
            if(i[0] in self.bnames):
                fig_bias = PLT.figure()
                TFT.display_matrix(np.array([i[1]]), fig=fig_bias, title=(i[0]))
                num2 += 1
                            
                            
    #The method for training the network                              
    def do_training(self,sess,cases,epochs=100,continued=False, steps=1):
        if not(continued): self.error_history = []
        #setting the variable mbs equal to the minibatch size, number of cases and number og minibatches. 
        mbs = self.minibatch_size; ncases = len(cases); nmb = math.ceil(ncases/mbs)

        #Epochs are how many times we run through the training set. 
        for st in range(0, steps): # Loop through cases, one minibatch at a time.
        
            error = 0; step = self.global_training_step + st
            gvars = [self.error] + self.grabvars
            
            #Picking out a number of random cases from the training cases where the number of cases is decided in the CONFIG file as minibatch size
            indices = np.random.randint(0,cases.shape[0],mbs)
            minibatch = cases[indices]
            
            #Steps is how many times we run through a new minibatch. If we run three steps with a minibatch of 10 we run through a total of 30 cases.
            
            #This is processing the minibatch to be ready to feed it into the network
            inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
            
            #Feeder that is put into the network with important information as input, target together with opmtimization function, probe and intervals and session
            feeder = {self.input: inputs, self.target: targets}
            _,grabvals,_ = self.run_one_step([self.trainer],gvars,self.probes,session=sess,
                                     feed_dict=feeder,step=step,show_interval=self.show_interval)
            
            #The error gotten from the network. The error is decided by the error function: either MBS or cross entropy.    
            error += grabvals[0]
            
            #Appending this error to the history of errors. 
            self.error_history.append((step, error/nmb))

            
            #Check if its time to run a validation set on the trainined network to check for overfitting
            self.consider_validation_testing(step,sess)
        #running the method to plot validation and training error
        self.plots()
        self.global_training_step += epochs
        #TFT.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
        #                          title="",fig=not(continued))

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.

    #Method for testing the network on a testing batch. Mostly the same as the training method, but with training turned off.
    def do_testing(self,sess,cases,msg='Testing',bestk=None):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor,[TFT.one_hot_to_int(list(v)) for v in targets],k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
                                           
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres/len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns an OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.
    
    #Checking to see which value from the output of the network is the highest and comparing this to the actual label to see if the netwwork correctly classified the case
    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))
        
    #Session for training including setting the session and probes as well as initializing the training on the training set
    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.roundup_probes() # this call must come AFTER the session is created, else graph is not in tensorboard.
        self.do_training(session,self.caseman.get_training_cases(),epochs,continued=continued, steps=self.config["steps"])
    
    #Much the same as the training session
    def testing_session(self,sess,bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing',bestk=bestk)
    
    #Same as testing and training session
    def mapping_session(self,sess,bestk=None):
        #cases = self.caseman.get_training_cases()
        cases = self.caseman.get_number_of_cases(self.config["Map Batch Size"])
        if len(cases) > 0:
            self.do_mapping(sess,cases)
    
    #Checking to see if its time for a validation run. If it is the error gotten from the validation is added to the validation history.
    def consider_validation_testing(self,epoch,sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess,cases,msg='Validation Testing')
                self.validation_history.append((epoch,error))


    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess,bestk=None):
        self.do_testing(sess,self.caseman.get_training_cases(),msg='Total Training',bestk=bestk)

    # Similar to the "quickrun" functions used earlier.
    #This is where the actual network is run. 
    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        
        
        #if show_interval and (step % show_interval == 0):
         #   self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1, labels=None):
        names = [x.name for x in grabbed_vars];
        ct = 0
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                fig_index += 1
            else:
                print(v, end="\n\n")
    
    
    
    def run(self,epochs=100,sess=None,continued=False,bestk=None):
        #PLT.ion()
        #This is the training of the network
        self.training_session(epochs,sess=sess,continued=continued)
        #This is running a test on the training set to see how many percentage correctly classifed cases we get. Traning turned off.
        self.test_on_trains(sess=self.current_session,bestk=bestk)
        #This is testing the network on the testing set with training turned off. 
        self.testing_session(sess=self.current_session,bestk=bestk)
        #This is the mapping session where I plot the maps and dedrogram
        self.mapping_session(sess=self.current_session, bestk=bestk)
        self.close_current_session(view=False)
        #PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).
#This is code to start up the network after training is done. Have not used this code. 
    def runmore(self,epochs=100,bestk=None):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True,bestk=bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self,view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)
    
    
    #Plotting of the training and validation error using the validation history and error history from training
    def plots(self):
        fig = PLT.figure()
        arr = np.array(self.validation_history)
        earr = np.array(self.error_history)
        steps = arr[:,0]
        values = arr[:,1]
        evalues = earr[:,1] 
        esteps = earr[:,0]


        
        dv = pd.DataFrame({'x': esteps, 'y': evalues})
        de = pd.DataFrame({'x': steps, 'y': values})
        concatenated = pd.concat([dv.assign(dataset='Training Error'), de.assign(dataset='Validation Error')])
        
        sns.lineplot(x='x', y='y', markers=True, dashes=False, hue="dataset", data=concatenated)
        
        # Set title
        PLT.title('Error')
        # Set x-axis label
        PLT.xlabel('Steps')
        # Set y-axis label
        PLT.ylabel('Error')
        fig.savefig('plots/graph')
        

# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self,ann,index,invariable,insize,outsize, config):
        self.ann = ann
        self.config = config
        self.insize=insize  # Number of neurons feeding into this module
        self.outsize=outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-"+str(self.index)
        self.build()
        

    def build(self):
        mona = self.name; n = self.outsize
        
#!!!        #These weights and biases should we set an upper and lower bound in the config file
        self.weights = tf.Variable(np.random.uniform(float(self.config["Initial Weight Range"][0]), float(self.config["Initial Weight Range"][1]), size=(self.insize,n)), name=mona+'-wgt',trainable=True) # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(float(self.config["Initial Weight Range"][0]), float(self.config["Initial Weight Range"][1]), size=n), name=mona+'-bias', trainable=True)  # First bias vector
        
        
#!!!        #This output should we be able to customize by setting in the config file.
        self.output = _hidden_activation_function[self.config["hidden activation function"]](tf.matmul(self.input,self.weights)+self.biases, name=mona+'-out')
        #self.output = tf.nn.relu(tf.matmul(self.input,self.weights)+self.biases,name=mona+'-out')
        self.ann.add_module(self)

    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self,type,spec):
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)
        
    


# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

    def __init__(self,cfunc,vfrac,tfrac, gen):
        self.gen = gen
        self.test = []
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        #If running on a dataset it choose from this dict and run the corresponding function
        self.datasets = {
            "wine": self.read_wine,
            "glass": self.read_glass,
            "yeast": self.read_yeast,
            "iris": self.read_iris,
            "mnist": self.read_mnist
        }
        
        if(gen):
            self.generate_cases()
        else:
            self.cases = []
            self.datasets[self.casefunc["data source"]]()
        self.organize_cases()
        
        
    #A method for normalizing the features of the dataset using standard deviation and mean.
    def normalize(self, arr):
        std = np.std(arr, axis=0)
        mean = np.mean(arr, axis=0)
        return (arr - mean)/std 


    #Depending on if the config says to use dataset or generate data. This is used when generating the different datasets
    def generate_cases(self):
        self.cases = _generator[self.casefunc["data source"]](*(self.casefunc["parameters"])) # Run the case generator.  Case = [input-vector, target-vector]
        if(self.casefunc["data source"] == "symmetry"):
            for i in range(len(self.cases)): 
                self.cases[i] = [self.cases[i][:-1],TFT.int_to_one_hot(int(self.cases[i][-1:][0]), 2)]
        
    
    #Reading the wine dataset, splitting it to input and target vectors and normalizing it    
    def read_wine(self, text_file="dataset/winequality_red.txt"):
        file_object = open(text_file, "r")
        k = np.genfromtxt(file_object, delimiter=";")
        x = k[:, :11]
        x = self.normalize(x).tolist()
        y = k[:, 11:].tolist()
        
        
        for i in range(len(x)):
            self.cases.append([x[i], TFT.int_to_one_hot(int(y[i][0])-3, 6)])

    #Reading in the Mnist dataset. 
    def read_mnist(self):
        cases, label = mnists.load_all_flat_cases()
        cases = np.array(cases)/255
        for i in range(int(len(cases))):
            self.cases.append([cases[i], TFT.int_to_one_hot(int(label[i]),10)])


    #Reading in the iris dataset. Otherwise same as wine
    def read_iris(self, text_file="dataset/iris.txt"):
        file_object = open(text_file, "r")
        k = np.genfromtxt(file_object, delimiter=",")
        x = k[:, :4]
        x = self.normalize(x).tolist()
        y = k[:, 4:].tolist()

        for i in range(len(x)):
            self.cases.append([x[i], TFT.int_to_one_hot(int(y[i][0]), 3)])
        
     #Reading in the yeast dataset. Otherwise same as wine
    def read_yeast(self, text_file="dataset/yeast.txt"):
        file_object = open(text_file, "r")
        k = np.genfromtxt(file_object, delimiter=",")
        x = k[:, :8]
        x = self.normalize(x).tolist()
        y = k[:, 8:].tolist()

        for i in range(len(x)):
            self.cases.append([x[i], TFT.int_to_one_hot(int(y[i][0])-1, 10)])
            
        
     #Reading in the glass dataset. Otherwise same as wine
    def read_glass(self, text_file="dataset/glass.txt"):
        file_object = open(text_file, "r")
        k = np.genfromtxt(file_object, delimiter=",")
        x = k[:, :9]
        x = self.normalize(x).tolist()
        y = k[:, 9:].tolist()

        for i in range(len(x)):
            self.cases.append([x[i], TFT.int_to_one_hot(int(y[i][0])-1, 7)])
        
        
    #Organizing the cases into training, testing and validation set   
    def organize_cases(self):
        fraction = round(len(self.cases) * float(self.casefunc["Case Fraction"]))
        self.cases = self.cases[0:fraction]
        ca = np.array(self.cases)    
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]
        
        
    #This is used for the mapping function to get a number of cases. They are different from testing, training and validation.
    def get_number_of_cases(self, size): 
        cl = np.array(self.cases)
        np.random.shuffle(cl)
        return cl[0:size]
        
    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases


#   ****  MAIN functions ****

# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
def autoex(epochs,nbits,lrate,showint,mbs,vfrac,tfrac,vint,sm,bestk, config, gen):
    mbs = mbs if mbs else size
    cman = Caseman(config, vfrac, tfrac, gen)
    ann = Gann(dims=config["dimesions"],cman=cman,lrate=lrate,showint=showint,mbs=mbs,vint=vint,softmax=sm, config=config)
    #ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    #ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    #ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    #ann.add_grabvar(0,'in')
    ann.run(epochs,bestk=bestk)
    
    # ann.runmore(epochs*2,bestk=bestk)

    #TFT.fireup_tensorboard('probeview')
    return ann

def countex(epochs=5000,nbits=15,ncases=500,lrate=0.5,showint=500,mbs=20,vfrac=0.1,tfrac=0.1,vint=200,sm=True,bestk=1):
    
    
    case_generator = (lambda: TFT.gen_vector_count_cases(ncases,nbits))
    cman = Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(dims=config["dimensions"], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint, softmax=sm)
    ann.run(epochs,bestk=bestk)
    TFT.fireup_tensorboard('probeview')
    return ann
    
    
#The main program intizializing the config file, deciding if it should generate or read and run the autoex function to run the program.
def main():
        generate = ["parity", "symmetry", "autoencoder", "bit counter", "segment counter"]
        gen = False
        with open("CONFIG.txt", 'r') as f:
                config = json.load(f)
                f.close()
                pprint(config)
        
        if(config["data source"] in generate):
            gen = True

        autoex(500, 4, config["learning rate"], 100, config["minibatch size"], config["validation fraction"], config["test fraction"], config["validation interval"], False, 1, config, gen)




if __name__ == '__main__':
        main()

