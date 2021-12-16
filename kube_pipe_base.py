import os
import yaml
import dill as pickle
import pandas as pd
import openapi_client
from openapi_client.api import workflow_service_api
from openapi_client.model.io_argoproj_workflow_v1alpha1_workflow_create_request import \
    IoArgoprojWorkflowV1alpha1WorkflowCreateRequest

import uuid
import glob
import functools
import atexit

import datetime
from dateutil.tz import tzutc

from time import sleep

from pprint import pprint

VOLUME_PATH = "/home/ansible/.kubetmp/"


def kubeconfig(resources):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs, resources=resources)
        return wrapper
    return decorator
        
class Kube_pipe_base:

    def __init__(self,*args):

        configuration = openapi_client.Configuration(host="http://192.168.30.240:2746", discard_unknown_keys=True)
        configuration.verify_ssl = False

        api_client = openapi_client.ApiClient(configuration)
        self.api = workflow_service_api.WorkflowServiceApi(api_client)

        self.pipelines = args

        self.pipeIds = []

        self.kuberesources = None

        self.functionresources = None

        atexit.register(self.deleteTemporaryFiles)


    def make_kube_pipeline(*args):
        return Kube_pipe_base(*args)


    def config(self, resources = None,function_resources = None):
        self.kuberesources = resources
        self.functionresources = function_resources


    def launchFromManifest(self,manifest):
        api_response = self.api.workflow_service_create_workflow(
            namespace='argo',
            body=IoArgoprojWorkflowV1alpha1WorkflowCreateRequest(
                workflow=manifest, _check_type=False,))
        
        name = api_response["metadata"]["name"]
        print(f"Launched workflow '{name}'")
        return name

    def deleteTemporaryFiles(self):
        for pipeid in self.pipeIds:
            files = glob.glob(f"{VOLUME_PATH}*{pipeid}.tmp")
            
            for f in files:
                os.remove(f)


    def waitForWorkflow(self,workflowName):

        while True:
            workflow = self.api.workflow_service_get_workflow(namespace="argo",name = workflowName)
            status = workflow["status"]

        
            if(getattr(status,"phase",None) is not None):

                if(status["phase"] == "Running"):
                    sleep(1)

                elif(status["phase"] == "Succeeded"):

                    endtime = datetime.datetime.now(tzutc())
                    starttime = workflow["metadata"]["creation_timestamp"]

                    print(f"\nWorkflow '{workflowName}' has finished. Time ({endtime-starttime})"u'\u2713')
                    return

                elif(status["phase"] == "Failed"):
                    raise Exception(f"Workflow {workflowName} has failed")


            print(".",end="",sep="",flush=True)


        



        