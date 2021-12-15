import os
import yaml
import pickle as pickle
import pandas as pd
import openapi_client
from openapi_client.api import workflow_service_api
from openapi_client.model.io_argoproj_workflow_v1alpha1_workflow_create_request import \
    IoArgoprojWorkflowV1alpha1WorkflowCreateRequest
import uuid

from kube_pipe_base import Kube_pipe_base

VOLUME_PATH = "/home/ansible/.kubetmp/"


def make_kube_pipeline(*args):
    return Kube_pipe(*args)

class Kube_pipe(Kube_pipe_base):

    def __init__(self,*args):
        super().__init__(*args)


    def workflow(self,X,y,funcs,name,pipeid,transformer,resources = None):

        with open(f'{VOLUME_PATH}X{pipeid}.tmp', 'wb') as handle:
                pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{VOLUME_PATH}y{pipeid}.tmp', 'wb') as handle:
                pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)


        with open(f"{os.path.dirname(os.path.realpath(__file__))}/yamls/workflow.yaml","r") as file:
            workflow = yaml.safe_load(file)


        templates = workflow["spec"]["templates"]
        workflow["metadata"]["generateName"] = name+str(pipeid)

        templates[0]["steps"] = []

        for i,func in enumerate(funcs):

            if getattr(func,"transform",None) is None and transformer is True:
                continue

            if(not transformer):
                with open(f'{VOLUME_PATH}func{i}{pipeid}.tmp', 'wb') as handle:
                    pickle.dump(func, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #Importar entradas
            code = f"""
import pickle

with open(\'X{pipeid}.tmp\', \'rb\') as input_file:
    X = pickle.load(input_file)
with open(\'y{pipeid}.tmp\', \'rb\') as input_file:
    y = pickle.load(input_file)

"""

            if( not transformer):
                code += f"""
with open(\'func{i}{pipeid}.tmp\', \'rb\') as input_file:
    func = pickle.load(input_file)

out = func.fit(X , y)
"""
            else:
                code += f"""
with open(\'transformer{i}{pipeid}.tmp\', \'rb\') as input_file:
    out = pickle.load(input_file)
"""
                
            if(getattr(func,"transform",None) is not None):
                code+=f"""

X = out.transform(X)

with open('X{pipeid}.tmp', \'wb\') as handle:
    pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('y{pipeid}.tmp', \'wb\') as handle:
    pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
                if not transformer:
                    code+=f"""                    
with open('transformer{i}{pipeid}.tmp', \'wb\') as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
            else:
                code+=f"""
with open('out{pipeid}.tmp', \'wb\') as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""    
            template = {'container': 
                            {'args': [''],
                            'command': ['python', '-c'],
                            'image': 'alu0101040882/scikit:p3.6.8',
                            'volumeMounts': [{'mountPath': '/usr/src/app',
                                            'name': 'workdir'}]},
                    'name': ''}

            if(resources is None):
                resources = self.kuberesources
            if(resources is not None):
                template["container"]["resources"]  = {"limits" : resources}

            template["name"] = str(i) + str.lower(str(type(func).__name__))
            template["container"]["args"][0] = code

            templates.append(template)


            templates[0]["steps"].append([{'name': template["name"],
                        'template': template["name"]}])

         
        return self.launchFromManifest(workflow)


    def fit(self,X,y, resources = None):
        
        self.pipeIds = []
        self.models = []

        workflowNames = []

        for i , pipeline in enumerate(self.pipelines):
            self.pipeIds.append(str(uuid.uuid4())[:8])

            workflowNames.append(self.workflow(X,y,pipeline,"workflow-fit-",self.pipeIds[i],False,resources = resources))

        for i, name in enumerate(workflowNames):
            self.waitForWorkflow(name)
            with open(f"{VOLUME_PATH}out{self.pipeIds[i]}.tmp","rb") as outfile:
                self.models.append(pickle.load(outfile))

        return self


    def score(self,X,y, pipeIndex = None):

        if self.pipelines == None or self.models == None:
            raise Exception("Model must be trained before calculating score")

        if pipeIndex == None:
            pipeIndex = range(len(self.pipelines))

        workflowNames = []

        scores = []

        
        for index in pipeIndex:
                workflowNames.append(self.workflow(X,y,self.pipelines[index][:-1],"workflow-score-",self.pipeIds[index],True))

        for i,index in enumerate(pipeIndex):
            self.waitForWorkflow(workflowNames[i])

            with open(f"{VOLUME_PATH}X{self.pipeIds[index]}.tmp","rb") as outfile:
                    testX = pickle.load(outfile)

            with open(f"{VOLUME_PATH}y{self.pipeIds[index]}.tmp","rb") as outfile:
                    testy = pickle.load(outfile)

            scores.append(self.models[index].score(testX,testy))

    
        return scores
        



        



        