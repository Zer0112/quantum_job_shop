from collections import namedtuple
from dataclasses import dataclass
from typing import Dict

#creates the classes for machines and jobs

@dataclass(frozen=True, repr=True)
class Machine:
    """Dataclass for the machine.
    It has a number to identify the machine, a name and a field for the number of
    parallel operations the machine can perform
    """
    nr: int
    name: str
    parallel_operations: int
    start_time: int = 0


class Machine_park:
    """The collection of all machines
    """

    def __init__(self):
        self.machines = {}
        self.nr_m = 0

    def add_machine(self, nr: int, name: str, parallel_operations=1, start=0):
        """adds a new machine to the park

        Args:
            nr (int): number of machine
            name (str): name of machine
            parallel_operations (int, optional): parallel operation the machine can perform. Defaults to 1.
        """
        self.machines.update(
            {nr: Machine(nr, name, parallel_operations, start)})
        self.nr_m += 1

    def generate_park(self, total_number: int):
        """Generates a machine park with a given number of machines

        Args:
            total_number (int): total number of machines in the park
        """
        for nr in range(total_number):
            self.add_machine(nr, f"m{nr}", 1)

    def __repr__(self):
        s = ""
        for x in self.machines.items():
            s += str(x)+"\n"
        return s

    @property
    def m(self):
        return self.machines


Workpackage = namedtuple('Workpackage', ['dauer', 'anzahl'])


class Job:
    """Job class with all the information to a job
    """

    def __init__(self, work: Dict, deadline: int = 0, id: int = 0, parallel_operations=1, start_time=0):
        """Job class constructor

        Args:
            work (Dict[Machine, Workpackage]): Mapping of Workpackage to Machine
            deadline (int): deadline of the job
            id (int): job id
        """

        self.id = id
        self.work = work
        self.deadline = deadline
        self.parallel_operations = parallel_operations
        self.start_time = start_time

    def add_workpackage(self, m: Machine, wp: Workpackage):
        """adds a workpackage to the job

        Args:
            m (Machine): machine on which the job has to do work
            wp (Workpackage): description of duration and repeation of the task
        """
        self.work.update({m: wp})

    def repeation_on_machine(self, m: Machine):
        try:
            _, r = self.work[m]
            return r
        except:
            return 0

    def duration_operation_machine(self, m: Machine):
        try:
            d, _ = self.work[m]
            return d
        except:
            return 0

    def machine_pool_job(self):
        return [m for m in self.work.keys()]

    def __repr__(self):
        s = f"Job {self.id} Operationen:\n"
        for m, op in self.work.items():
            durations, times = op
            s += f"{m} Dauer: {durations} Anzahl: {times} \n"

        if self.deadline != 0:
            s += f"deadline: {self.deadline}"
        if self.parallel_operations != 1:
            s += f"parallel operations: {self.parallel_operations}"
        return s
