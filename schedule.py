
from job import Job, Machine, Machine_park

from datetime import date, timedelta
import pandas as pd
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass
from typing import DefaultDict, Dict
from qubo_calc import *
import plotly.express as px
import numpy as np
# for making sure python still works even if some of the visualization packages are not installed
try:
    from holoviews import opts
    import hvplot.networkx as hvnx
    import networkx as nx
    import holoviews as hv
    hv.extension('bokeh')
    defaults = dict(width=400, height=400)
    hv.opts.defaults(
        opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
except:
    print("Connected Qubo visualization is not available")

Workpackage = namedtuple('Workpackage', ['dauer', 'anzahl'])
# Machine = str


class Schedule:
    """Schedule does the bookkeeping of all jobs and machines.
    """

    def __init__(self, machine_park: Machine_park = Machine_park(), time_max=10):
        self.machine_park = machine_park

        self.workload = OrderedDict()

        self.time_max = time_max
        # self.nr_machines = machine_park.nr_m
        self.nr_jobs = 0

    def build_machines(self, nr_machines):
        """Does generate nr_machines x machines
        """
        self.machine_park.generate_park(nr_machines)

    def add_machine(self, parallel_operations, start=0):
        """Adds one machine

        Args:
            parallel_operations ([type]):
            start (int, optional): earliest possible start because of initial conditions. Defaults to 0.
        """
        nr = self.machine_park.nr_m
        self.machine_park.add_machine(nr, f"m{nr}", parallel_operations, start)

    def add_job(self, job: Job):
        """adds one job
        """
        self.nr_jobs += 1
        self.workload.update({job.id: job})

    def create_job(self, work_per_m, deadline=0, parallel_operations=1, start=0):
        """Creates one job

        Args:
            work_per_m ([type]): operation dict mapping the work to machines
            deadline (int, optional):  Defaults to 0 means no deadline other than the global deadline.
            parallel_operations (int, optional): nr of parallel operations. Defaults to 1.
            start (int, optional): earliest possible start
        """
        id1 = self.nr_jobs
        dict_work = {self.machine_park.m[a]: b for a, b in work_per_m.items()}
        job = Job(dict_work, deadline, id1, parallel_operations, start)
        self.add_job(job)

    def __repr__(self):
        st = "---------------------------- \n"
        st += "Arbeitsplan\n"
        for x in self.workload:
            st += str(self.workload[x])+"\n"

        st += "---------------------------- \n"
        return st

    @property
    def job(self):
        return self.workload

    def get_time_jm_per_o(self, j_id: int, m: Machine):
        """Time per task of jop=j_id on machine m
        """
        try:
            work_job = self.workload[j_id].work
            time, _ = work_job[m]
        except:
            time, _ = 0, 0
        return time

    def get_time_jm_total(self, j_id: int, m: Machine):
        """total Time of the work of jop=j_id on machine m
        """
        try:
            work_job = self.workload[j_id].work
            time, times = work_job[m]
        except:
            time, times = 0, 0
        return time*times

    def get_times_of_operation(self, j_id: int, m: Machine):
        """Calculates how often a certain task from job=j_id has to be executed
        """
        try:
            work_job = self.workload[j_id].work
            _, times = work_job[m]
        except KeyError:
            times = 0
        return times


@dataclass(frozen=True, order=True)
class State:
    """State class is for describing a state in the execution of one full workload. 
    e.g. (job1,machine1,time1) - job1 is on machine1 at time=time1
    """
    job: Job
    machine: Machine
    time: int
    virtual_job: int = 0
    virtual_machine: int = 0
    virtuell: bool = False

    def __repr__(self):
        if self.virtuell == False:
            return f"state(j:{self.job.id},m:{self.machine.nr},t:{self.time})"
        else:
            return f"state(j:{(self.job.id, self.virtual_job)},m:{(self.machine.nr, self.virtual_machine)},t:{self.time})"

    def to_tuple(self):
        if self.virtuell == False:
            t = (self.job.id, self.machine.nr, self.time)
        else:
            t = ((self.job.id, self.virtual_job),
                 (self.machine.nr, self.virtual_machine), self.time)
        return tuple(t)


class Qubo:
    """Class for calculating Qubo
    """

    def __init__(self, s: Schedule):
        self.schedule = s

        # todo
        self.penalty_terms = {x: 1 for x in range(0, 8)}

        # a machine can only work on one job only at a time
        # a machine can for example not work on job 1 and job 2 at the same time

        # one operation/product can only be at one machine (can not be splitted to get produced at 2 or more machines at the same time)
        self.one_machine_per_operation = True
        self.h = OrderedDict()
        self.state_id = OrderedDict()
        self.J = OrderedDict()
        self.J_terms = [defaultdict(int) for _ in range(8)]

        self.nr_to_s_mapping = {}
        self.s_to_nr_mapping = {}

        self.solution_data = None
        self.test_energy = 0

    def jobs(self):
        return self.workload.values()

    def maschines(self):
        return self.schedule.machine_park.m.values()

    def calculate_ising(self, qubo):
        # gives the ising back from Qubo and without offset
        J_ising = {(s1, s2): v/4 for (s1, s2), v in qubo.items() if s1 != s2}
        h_ising = {(s1): -1/2*(sum(v for (z1, _), v in qubo.items() if z1 == s1)
                               + sum(v for (_, z2), v in qubo.items() if z2 == s1))
                   for (s1, s2), v in qubo.items() if s1 == s2}
        return (J_ising, h_ising)

    def update_J(self, s1: State, s2: State):
        """calculates off diagonal Qubo

        Args:
            s1 (State)
            s2 (State)

        Returns:
            Q[s1,s2]
        """
        j1 = s1.job.id
        j2 = s2.job.id
        m1 = s1.machine.nr
        m2 = s2.machine.nr
        t1 = s1.time
        t2 = s2.time
        temp = 0
        tmax = self.schedule.time_max
        p = self.penalty_terms
        j1_deadline, j1_parallel = self.find_parameter_j(tmax, s1.job)
        (t1_jm_start, parallel_operations_m1, repeat_operation,
         duration_operation_machine) = self.find_parameter_jm(s1.job, s1.machine)
        if s1.virtuell == True:
            vm1 = s1.virtual_job
            vm2 = s2.virtual_job
            vj1 = s1.virtual_machine
            vj2 = s2.virtual_machine
            p1 = calculate_qubo_H1_J(j1, m1, t1, j2, m2,
                                     t2)
            p2 = calculate_qubo_H2_J(j1, m1, t1,
                                     j2, m2, t2, duration_operation_machine, vm1=vm1, vm2=vm2, vj1=vj1, vj2=vj2)
            p3 = calculate_qubo_H3_J(j1, m1, t1,
                                     j2, m2, t2, duration_operation_machine, vm1=vm1, vm2=vm2, vj1=vj1, vj2=vj2)
            p4 = calculate_qubo_H4_J(j1, m1, t1, j2, m2, t2)
        else:
            p1 = calculate_qubo_H1_J(j1, m1, t1, j2, m2,
                                     t2)
            p2 = calculate_qubo_H2_J(j1, m1, t1,
                                     j2, m2, t2, duration_operation_machine)
            p3 = calculate_qubo_H3_J(j1, m1, t1,
                                     j2, m2, t2, duration_operation_machine)
            p4 = calculate_qubo_H4_J(j1, m1, t1, j2, m2, t2)

        temp += p[1]*p1
        self.J_terms[1][(s1, s2)] += p[1]*p1

        temp += p[2]*p2
        self.J_terms[2][(s1, s2)] += p[2]*p2

        temp += p[3]*p3
        self.J_terms[3][(s1, s2)] += p[3]*p3

        temp += p[4]*p4
        self.J_terms[4][(s1, s2)] += p[4]*p3

        # no off diagonal terms
        p5 = 0
        temp += p[5]*p5
        return temp

    def update_h(self, s1: State):
        """calculates diagonal Qubo

        Args:
            s1 (State)

        Returns:
            Q[s1,s1]
        """
        j1 = s1.job.id
        m1 = s1.machine.nr
        t1 = s1.time
        p = self.penalty_terms
        temp = 0
        (t1_jm_start, parallel_operations_m1, repeat_operation,
         duration_operation_machine) = self.find_parameter_jm(s1.job, s1.machine)
        if s1.virtuell == True:
            vm1 = s1.virtual_job
            vj1 = s1.virtual_machine
            p1 = calculate_qubo_H1_h(j1, m1, t1, repeat_operation)
            p2 = calculate_qubo_H2_h(j1, m1, t1, )
            p3 = calculate_qubo_H3_h(j1, m1, t1, )
            p4 = calculate_qubo_H4_h(j1, m1, t1, )
            p5 = calculate_qubo_H5_h(j1, m1, t1,
                                     s1.job.deadline, self.schedule.time_max)
        else:
            p1 = calculate_qubo_H1_h(j1, m1, t1, repeat_operation)
            p2 = calculate_qubo_H2_h(j1, m1, t1, )
            p3 = calculate_qubo_H3_h(j1, m1, t1, )
            p4 = calculate_qubo_H4_h(j1, m1, t1, )
            p5 = calculate_qubo_H5_h(j1, m1, t1,
                                     s1.job.deadline, self.schedule.time_max)

        temp += p[1]*p1
        self.J_terms[1][(s1, s1)] += p[1]*p1

        temp += p[2]*p2
        self.J_terms[2][(s1, s1)] += p[2]*p2

        temp += p[3]*p3
        self.J_terms[3][(s1, s1)] += p[3]*p3

        temp += p[4]*p4
        self.J_terms[4][(s1, s1)] += p[4]*p4

        temp += p[5]*p5
        self.J_terms[5][(s1, s1)] += p[5]*p5

        return temp

    def calculate_qubo(self):
        """Calculates the full Qubomatrix in triangle form. s2>s1 
        """
        self.test_energy
        M = self.schedule.machine_park.m.values()
        J = self.schedule.workload.values()
        t_max = self.schedule.time_max
        # j_max = self.schedule.nr_jobs
        # m_max = self.schedule.nr_machines
        M1 = M
        M2 = M
        self.h = OrderedDict()
        self.J = DefaultDict(int)

        for j1 in J:
            (job_deadline1, parallel_operations_j1) = self.find_parameter_j(t_max, j1)
            # for pj1 in range(parallel_operations_j1):
            # M1 = j1.machine_pool_job()
            for m1 in M1:
                (t1_jm_start, parallel_operations_m1, repeat_operation,
                 duration_operation_machine) = self.find_parameter_jm(j1, m1)
                # for pm1 in range(parallel_operations_m1):
                self.test_energy += repeat_operation**2
                for t1 in range(t1_jm_start, job_deadline1-duration_operation_machine+1):
                    for j2 in J:
                        if j1.id > j2.id:
                            continue
                        (job_deadline2, parallel_operations_j2) = self.find_parameter_j(
                            t_max, j2)
                        # for pj2 in range(parallel_operations_j2):
                        # M2 = j2.machine_pool_job()
                        for m2 in M2:

                            if m1.nr > m2.nr:
                                continue
                            (t2_jm_start, parallel_operations_m2, repeat_operation2,
                             duration_operation_machine2) = self.find_parameter_jm(j2, m2)

                            # for pm1 in range(parallel_operations_m1):
                            for t2 in range(t1, job_deadline2-duration_operation_machine2+1):

                                st1 = State(
                                    j1, m1, t1)
                                st2 = State(
                                    j2, m2, t2)
                                if (st1 == st2):
                                    t = self.update_h(st1)
                                    self.h.update({st1: t})
                                    self.state_id.update(
                                        {(j1.id, m1.nr, t1): st1})
                                    self.J.update(
                                        {(st1, st1): t})
                                else:
                                    self.J.update(
                                        {(st1, st2): self.update_J(st1, st2)})
        self.generate_mapping()

    def calculate_qubo_virtuell(self):
        """Calculates the full Qubomatrix in triangle form. s2>s1 with virtual jobs/machines for parallel operations
        """
        M = self.schedule.machine_park.m.values()
        J = self.schedule.workload.values()
        t_max = self.schedule.time_max
        # j_max = self.schedule.nr_jobs
        # m_max = self.schedule.nr_machines
        M1 = M
        M2 = M
        self.h = OrderedDict()
        self.J = DefaultDict(int)
        self.state_id = OrderedDict()
        # loop over all possible states
        # hard to make it more readable
        # it just means
        # for all states1:
        # for all states2>states1:
        for j1 in J:
            (job_deadline1, parallel_operations_j1) = self.find_parameter_j(t_max, j1)
            for vj1 in range(parallel_operations_j1):
                # M1 = j1.machine_pool_job()
                for m1 in M1:
                    (t1_jm_start, parallel_operations_m1, repeat_operation,
                     duration_operation_machine) = self.find_parameter_jm(j1, m1)
                    if vj1 == 0:
                        self.test_energy += repeat_operation**2
                    for vm1 in range(parallel_operations_m1):
                        for t1 in range(t1_jm_start, job_deadline1-duration_operation_machine+1):
                            for j2 in J:
                                if j1.id > j2.id:
                                    continue
                                (job_deadline2, parallel_operations_j2) = self.find_parameter_j(
                                    t_max, j2)
                                for vj2 in range(vj1, parallel_operations_j2):
                                    # M2 = j2.machine_pool_job()
                                    for m2 in M2:

                                        if m1.nr > m2.nr:
                                            continue
                                        (t2_jm_start, parallel_operations_m2, repeat_operation2,
                                         duration_operation_machine2) = self.find_parameter_jm(j2, m2)

                                        for vm2 in range(vm1, parallel_operations_m1):
                                            for t2 in range(t1, job_deadline2-duration_operation_machine2+1):

                                                st1 = State(
                                                    j1, m1, t1, virtual_job=vj1, virtual_machine=vm1, virtuell=True)
                                                st2 = State(
                                                    j2, m2, t2, virtual_job=vj2, virtual_machine=vm2, virtuell=True)
                                                if (st1 == st2):
                                                    t = self.update_h(st1)
                                                    self.h.update({st1: t})
                                                    self.state_id.update(
                                                        {(j1.id, m1.nr, t1): st1})
                                                    self.J.update(
                                                        {(st1, st1): t})
                                                else:
                                                    self.J.update(
                                                        {(st1, st2): self.update_J(st1, st2)})
        self.generate_mapping()

    def find_parameter_j(self, t_max, j1):
        parallel_operations_j1 = j1.parallel_operations
        job_deadline = t_max
        if j1.deadline != 0:
            job_deadline = j1.deadline
        elif j1.deadline > t_max:
            raise ValueError(
                "the deadline of this job is bigger than the total deadline, if that is intended then set the deadline of the job to 0")
        return (job_deadline, parallel_operations_j1)

    def find_parameter_jm(self, j1: Job, m1: Machine):
        parallel_operations_m1 = m1.parallel_operations
        repeat_operation = j1.repeation_on_machine(m1)
        duration_operation_machine = j1.duration_operation_machine(m1)
        t1_jm_start = max(j1.start_time, m1.start_time)
        return (t1_jm_start, parallel_operations_m1, repeat_operation, duration_operation_machine)

    def generate_mapping(self):
        """generates mapping from states to int and int to states from a qubo
        """
        mapping2 = {y: x for x, y in enumerate(self.h)}
        mapping1 = {x: y for x, y in enumerate(self.h)}
        self.nr_to_s_mapping = mapping1
        self.s_to_nr_mapping = mapping2

    def plot_qubo(self):
        """Plots the Qubo as heatmap
        """
        if not self.s_to_nr_mapping:
            self.generate_mapping()
        mapping = self.s_to_nr_mapping
        l = len(self.h)
        mat1 = Qubo.create_np_J(self.J, mapping, l)
        fig1 = px.imshow(mat1)
        # fig1.show()
        return fig1

    def plot_dict_matrix(dict_j, l):
        """Heatmap of the qubo"""
        mat1 = np.zeros((l, l))
        for s, value in dict_j.items():
            mat1[s[0]][s[1]] = value
        fig1 = px.imshow(mat1)
        # fig1.show()
        return fig1

    def create_np_J(dict_j, mapping, l):
        """Helper for heatmap"""
        q1 = {}
        for (a, b), x in dict_j.items():
            q1.update({(mapping[a], mapping[b]): x})
        mat1 = np.zeros((l, l))
        for s, value in q1.items():
            mat1[s[0]][s[1]] = value
        return mat1

    def plot_connections_qubo(self, size=400):
        """Plots the qubo with nr as labels"""
        G2 = nx.Graph()
        map1 = self.s_to_nr_mapping
        for (s1, s2), w in (self.J.items()):
            if w == 0:
                continue
            if s1 == s2:
                continue
            else:
                G2.add_edge(map1[s1], map1[s2], weight=w)
        for i, s in self.h.items():
            G2.add_node(map1[i], size=s)
        g = hvnx.draw(G2, nx.layout.circular_layout, edge_color='weight', edge_cmap='coolwarm',
                      edge_width=hv.dim('weight'), node_color="size", node_cmap='coolwarm', height=size, width=size)
        g1 = hvnx.draw(G2, nx.layout.spring_layout, edge_color='weight', edge_cmap='coolwarm',
                       edge_width=hv.dim('weight'), node_color="size", node_cmap='coolwarm', height=size, width=size)
        g2 = hvnx.draw(G2, nx.layout.kamada_kawai_layout, edge_color='weight', edge_cmap='coolwarm',
                       edge_width=hv.dim('weight'), node_color="size", node_cmap='coolwarm', height=size, width=size)
        return g+g1+g2

    def plot_connections_qubo_states(self, size=400):
        """Plots the qubo with states as labels
        """
        G2 = nx.Graph()
        g = self.plot_general_graph(size, G2, self.J)
        g1 = hvnx.draw(G2, nx.layout.spring_layout, edge_color='weight', edge_cmap='coolwarm',
                       edge_width=hv.dim('weight'), node_color="size", node_cmap='coolwarm', height=size, width=size)
        g2 = hvnx.draw(G2, nx.layout.kamada_kawai_layout, edge_color='weight', edge_cmap='coolwarm',
                       edge_width=hv.dim('weight'), node_color="size", node_cmap='coolwarm', height=size, width=size)
        return g+g1+g2

    def plot_general_graph(self, size, G2, dic_graph, layout=nx.layout.circular_layout):
        """Plot helper function
        """
        for (s1, s2), w in (dic_graph.items()):
            if w == 0:
                continue
            if s1 == s2:
                continue
            else:
                t1 = s1.to_tuple()
                t2 = s2.to_tuple()
                G2.add_edge(t1, t2, weight=w)
        for i, s in self.h.items():
            G2.add_node(i.to_tuple(), size=s)
        g = hvnx.draw(G2, layout, edge_color='weight', edge_cmap='coolwarm',
                      edge_width=hv.dim('weight'), node_color="size", node_cmap='coolwarm', height=size, width=size)

        return g

    def plot_qubo_terms(self, size=400):
        """Plots the contributions of the penalty terms 1-3
        """
        G = nx.Graph()
        g1 = self.plot_general_graph(size, G, self.J_terms[1])
        g2 = self.plot_general_graph(size, G, self.J_terms[2])
        g3 = self.plot_general_graph(size, G, self.J_terms[3])
        g1l = self.plot_general_graph(
            size, G, self.J_terms[1], nx.layout.spring_layout)
        g1l2 = self.plot_general_graph(
            size, G, self.J_terms[1], nx.layout.kamada_kawai_layout)
        g2l = self.plot_general_graph(
            size, G, self.J_terms[2], nx.layout.spring_layout)
        g2l2 = self.plot_general_graph(
            size, G, self.J_terms[2], nx.layout.kamada_kawai_layout)
        g3l = self.plot_general_graph(
            size, G, self.J_terms[3], nx.layout.spring_layout)
        g3l2 = self.plot_general_graph(
            size, G, self.J_terms[3], nx.layout.kamada_kawai_layout)
        return hv.Layout(g1+g2+g3+g1l+g2l+g3l+g1l2+g2l2+g3l2).cols(3)

    def create_panda_df(self):
        """create panda dataframe of qubo"""
        if not self.J:
            self.calculate_qubo()
        j_dic = self.J
        mapping = self.s_to_nr_mapping
        data1 = [dict(s1=mapping[s1], s1_ob=s1, j1_m1_t1=(s1.job.id, s1.machine.nr, s1.time),
                      s2=mapping[s2], s2_ob=s2, j2_m2_t2=(
            s2.job.id, s1.machine.nr, s2.time),
            value=v) for (s1, s2), v in j_dic.items()]
        df = pd.DataFrame(data1)
        return df

    def interpret_solution(self, solution_state):
        """Legacy code if we get the solution as bitstring
        """
        self.solution_data = None
        lst = list(solution_state)
        solution_nr = [x for x, y in enumerate(lst) if int(y) == 1]
        solution_state = [self.nr_to_s_mapping[x] for x in solution_nr]

        init_date = date(2021, 1, 1)
        day = timedelta(days=1)

        dict_solution = []
        for s in solution_state:
            j = s.job
            m = s.machine
            start = init_date + day * s.time
            end = start + j.duration_operation_machine(m)*day
            dict_solution.append(
                dict(job_id=str(j.id), machine_nr=str(m.nr), start=start, end=end, state=s, job=j, machine=m))

        df = pd.DataFrame(dict_solution)
        energy = sum(self.J[(x, x)] for x in solution_state) + \
            sum(self.J[(x, y)]
                for x in solution_state for y in solution_state if x != y)

        energy_term = self.calculate_energy_terms(solution_state)

        self.solution_data = (solution_nr, solution_state, energy, df)
        return (solution_nr, solution_state, energy, energy_term, self.test_energy, df)

    def calculate_energy_terms(self, solution_state):
        energy_term = dict()

        for i in range(1, 6):
            energy_term[i] = sum((self.J_terms[i])[(x, x)] for x in solution_state) + \
                sum((self.J_terms[i])[(x, y)]
                    for x in solution_state for y in solution_state if x != y)

        return energy_term

    def interpret_solution_dict(self, solution_state):
        """Interprets the response of the solver and sets it as the solution to the qubo. 
        It calculates the energy of the solution, the contribution of single penalty terms
        and the absolute minimum of the energy

        Args:
            solution_state : solution states of the solver (best sample)

        Returns:
            (solution_nr, solution_state, energy, energy_term, self.test_energy, dataframe of solution with start- and endtime)
        """
        self.solution_data = None
        mapping = self.s_to_nr_mapping
        solution_nr = [mapping[x] for x, y in solution_state.items() if y == 1]
        solution_state = [x for x, y in solution_state.items() if y == 1]

        init_date = date(2021, 1, 1)
        day = timedelta(days=1)

        dict_solution = []
        for s in solution_state:
            j = s.job
            m = s.machine
            start = init_date + day * s.time
            end = start + j.duration_operation_machine(m)*day
            dict_solution.append(
                dict(job_id=str(j.id), machine_nr=str(m.nr), start=start, end=end, state=s, job=j, machine=m))

        df = pd.DataFrame(dict_solution)
        energy = sum(self.J[(x, x)] for x in solution_state) + \
            sum(self.J[(x, y)]
                for x in solution_state for y in solution_state if x != y)

        energy_term = self.calculate_energy_terms(solution_state)

        self.solution_data = (solution_nr, solution_state, energy, df)
        return (solution_nr, solution_state, energy, energy_term, self.test_energy, df)

    def plot_solution(self):
        """grantt of job vs time"""
        if not self.solution_data:
            print("No solution data")
            print("Please use interpret_solution first")
            return
        *_, df = self.solution_data
        fig = px.timeline(df, x_start="start", x_end="end",
                          y="job_id", color="machine_nr")
        return fig

    def plot_solution2(self):
        """grantt of machine vs time"""
        if not self.solution_data:
            print("No solution data")
            print("Please use interpret_solution first")
            return
        *_, df = self.solution_data
        fig = px.timeline(df, x_start="start", x_end="end",
                          y="machine_nr", color="job_id")
        return fig


if __name__ == '__main__':
    mp1 = Machine_park()
    mp1.generate_park(3)
    j1_m1 = Workpackage(dauer=1, anzahl=1)
    j1_m2 = Workpackage(dauer=3, anzahl=1)
    work1 = {mp1.machines[1]: j1_m1, mp1.machines[2]: j1_m2}
    deadline1 = 6
    job_id = 1
    j1 = Job(work1, deadline1, job_id)
    j1.add_workpackage(mp1.machines[2], Workpackage(1, 2))
    print(j1)

    s = Schedule(mp1, time_max=5)
    s.add_job(j1)

    work2 = {mp1.machines[1]: (1, 1), mp1.machines[2]: (3, 1)}
    deadline2 = 8
    j2 = Job(work2, deadline2, 2)
    s.add_job(j2)
    print("time of job 1 on machine 1 = ",
          s.get_time_jm_per_o(1, mp1.machines[1]))

    mp = Machine_park()
    mp.add_machine(nr=1, name="m2", parallel_operations=1)
    mp.generate_park(3)
    print("nr_jobs", s.nr_jobs)
    print(mp)
    print(mp.machines[1])

    qubo = Qubo(s)
    print("Qubo")
    qubo.calculate_qubo()
    counter = 0
    for x in qubo.J.items():
        counter += 1
        print(x)
        if counter > 3:
            print("....")
            break
    counter = 0
    for x in enumerate(qubo.h):
        counter += 1
        print(x)
        if counter > 3:
            print("....")
            break
    qubo.generate_mapping()
    # print(qubo.nr_to_s_mapping, "\n")
    # print(qubo.s_to_nr_mapping)
    f = qubo.plot_qubo()
    d = qubo.create_panda_df()
    print(d)
    print(qubo.interpret_solution([1, 1, 1, 1, 1, 0, 1, 1, 1, 0]))
    qubo.calculate_ising(qubo.J)
    # f.show()
