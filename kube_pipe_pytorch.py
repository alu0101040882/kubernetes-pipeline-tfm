import os
import yaml

import cloudpickle as pickle

from pickle import DEFAULT_PROTOCOL as protocol

import uuid

from kube_pipe_base import Kube_pipe_base

VOLUME_PATH = "/home/ansible/.kubetmp/"

def make_kube_pipeline(*args):
    return Kube_pipe(*args)

def make_kube_pipeline2(*args):
    arguments = []

    for arg in args:
        processedarg = {}
        processedarg["loss_fn"] = arg[0]
        processedarg["train_fn"] = arg[1]
        processedarg["test_fn"] = arg[2]
        processedarg["optimizer"] = arg[3]
        processedarg["model"] = arg[4]

        arguments.append(processedarg)


    return Kube_pipe(*arguments)

class Kube_pipe(Kube_pipe_base):

    def __init__(self,*args):
        super().__init__(*args)

        self.trainings = args

        self.paramkeys = ["train_fn","test_fn","loss_fn","optimizer","model"]

    def workflow(self,train_data,test_data,training,name,pipeid,epochs,resources = None):

        with open(f'{VOLUME_PATH}train_data{pipeid}.tmp', 'wb') as handle:
            pickle.dump(train_data, handle,protocol=protocol)

        with open(f'{VOLUME_PATH}test_data{pipeid}.tmp', 'wb') as handle:
            pickle.dump(test_data, handle,protocol=protocol)


        for key in self.paramkeys:
            with open(f'{VOLUME_PATH}{key}{pipeid}.tmp', 'wb') as handle:
                pickle.dump(training[key], handle,protocol=protocol)
  
        
        #save(training["model"], f'{VOLUME_PATH}model{pipeid}.tmp')

        with open(f'{VOLUME_PATH}modelclass{pipeid}.tmp', 'wb') as handle:
            pickle.dump(training["model"].__class__, handle)

        with open(f"{os.path.dirname(os.path.realpath(__file__))}/yamls/workflow.yaml","r") as file:
            workflow = yaml.safe_load(file)

        templates = workflow["spec"]["templates"]
        workflow["metadata"]["generateName"] = name+str(pipeid)

        templates[0]["steps"] = []

            #Importar entradas
        code = f"""

import cloudpickle as pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

print(pickle)

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(\'train_data{pipeid}.tmp\', \'rb\') as input_file:
    train_data = pickle.load(input_file)

with open(\'test_data{pipeid}.tmp\', \'rb\') as input_file:
    test_data = pickle.load(input_file)

"""
        for key in self.paramkeys:
            code+=f"""
with open(\'{key}{pipeid}.tmp\', \'rb\') as input_file:
    {key} = pickle.load(input_file)
    print('{key} loaded')

"""

        code+=f"""
print("Datos cargados")

model.eval()

print("Modelo cargado")

print("Traindata", train_data)
print("Trainfn", train_fn)

epochs = {epochs}
for t in range(epochs):
    print("Epoch ", t+1)
    train_fn(train_data, model, loss_fn, optimizer)
    
    test_fn(test_data, model, loss_fn)
print("Done!")


with open(f'out{pipeid}.tmp', 'wb') as handle:
    pickle.dump(model, handle)

"""

        template = {'container': 
                            {'args': [''],
                            'command': ['python', '-c'],
                            'image': 'alu0101040882/pytorch:p3.6.8',
                            'volumeMounts': [{'mountPath': '/usr/src/app',
                                            'name': 'workdir'}]},
                    'name': ''}

        template["name"] = pipeid
        template["container"]["args"][0] = code

        if(resources is None):
            resources = self.kuberesources

        if(resources is not None):
            template["container"]["resources"]  = {"limits" : resources}

        templates.append(template)


        templates[0]["steps"].append([{'name': template["name"],
                    'template': template["name"]}])


        return self.launchFromManifest(workflow)

    def score(self, dataloader,index = None):
        if(index != None):
            return self.trainings[index]["test_fn"](dataloader,self.models[index],self.trainings[index]["loss_fn"])


        out = []
        for index in range(len(self.models)):
            out.append(self.trainings[index]["test_fn"](dataloader,self.models[index],self.trainings[index]["loss_fn"]))

        return out



    def getModel(self, trainIndex):
        return self.models[trainIndex]

    def train(self,train_data,test_data, epochs = 5, kuberesources = None):
        
        self.pipeIds = []
        self.models = []

        workflowNames = []

        for i , training in enumerate(self.trainings):
            self.pipeIds.append(str(uuid.uuid4())[:8])
            workflowNames.append(self.workflow(train_data,test_data,training,"workflow-train-",self.pipeIds[i],epochs,resources=kuberesources))

        for i, name in enumerate(workflowNames):
            self.waitForWorkflow(name)
            with open(f"{VOLUME_PATH}out{self.pipeIds[i]}.tmp","rb") as outfile:
                self.models.append(pickle.load(outfile))

        return self





        



        