import threading
from datetime import datetime
from copy import copy


class State:
    def __init__(self):
        self.exit_event = threading.Event()
        self.text = ""

    @property
    def data(self):
        data = copy(self.text)
        data['ts'] = datetime.utcnow().timestamp()
        return data
