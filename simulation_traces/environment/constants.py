from model_apps import apps
from model_infra import Infra


TIME_PERIOD = 1
N_PM = Infra().getInfraSize()
N_APPS = len(apps)

LAMBDA = 0
LAMBDA_2 = 10