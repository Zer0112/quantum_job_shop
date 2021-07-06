import datetime
import pickle
import dimod
from dimod.reference.samplers.exact_solver import ExactSolver
from dwave.system import DWaveSampler, EmbeddingComposite
import neal
from dwave_qbsolv import QBSolv
try:
    from uqoclient.client.connection import Connection
    from uqoclient import Problem
    from uqoclient.client.config import Config
    config = Config(configpath="config.json")
    connection = config.create_connection()
except:
    print("no uqo")
from dwave.system import DWaveSampler, EmbeddingComposite
try:
    solver = DWaveSampler(solver=dict(qpu=True))
    samplerq = EmbeddingComposite(solver)
except:
    print("no direct access to dwave")


def solve_uqo(Q, samples=100):
    return Problem.Qubo(config, Q).with_platform("qbsolv").solve(samples)


def solve_dwave(Q, samples=100):
    problem = Problem.Qubo(config, Q).with_platform(
        "dwave").with_solver("Advantage_system1.1")
    answer = problem.solve(samples)
    return (problem, answer)


def find_embedding(Q):
    sampler_auto = EmbeddingComposite(
        DWaveSampler(solver={'topology__type': 'chimera'}))


def solve_dau(Q, samples=100):
    problem = Problem.Qubo(config, Q).with_platform(
        "fujitsu").with_solver("DAU")
    answer = problem.solve(samples)
    return (problem, answer)


def solve_simulated_annealing(Q, samples=100):
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=samples)
    return sampleset


def solve_qbsolv(Q, samples=100):
    response = QBSolv().sample_qubo(Q, num_reads=samples)
    return response


def solvelog_dwave(qubo, samples=100):
    J = qubo.J
    s = qubo.schedule
    ts = int(datetime.datetime.now().timestamp()*1000)
    name = f"{ts}_{len(s.workload)}-{s.machine_park.nr_m}-{s.time_max}-{len(qubo.h)}"
    pickle.dump(qubo, open(f"./logs/{name}.p", "wb"))
    answer = samplerq.sample_qubo(
        {x: y for x, y in J.items() if y != 0}, num_reads=samples, embedding_parameters=dict(timeout=600))
    pickle.dump(answer, open(f"./logs/{name}_response.p", "wb"))
    return answer


if __name__ == "__main__":
    pass
