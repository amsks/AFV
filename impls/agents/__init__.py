from agents.afv import AFVAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.idm import IDMAgent
from agents.qrl import QRLAgent

agents = dict(
    afv=AFVAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    idm=IDMAgent,
    qrl=QRLAgent,
)
