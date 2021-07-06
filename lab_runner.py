from copy import deepcopy
import pickle
from schedule import *
import datetime
from anneal_solver import *
from os import listdir
from os.path import isfile, join
# s = Schedule(time_max=5)
# s.build_machines(1)
# s.create_job({0: (1, 1)}, parallel_operations=1)


def dump_qubo(s):
    # print(f"nr_j={len(s.workload)},nr_m={s.machine_park.nr_m},tmax={s.time_max}")
    ts = int(datetime.datetime.now().timestamp()*1000)
    name = f"qubo_s_{len(s.workload)}-{s.machine_park.nr_m}-{s.time_max}_{ts}"
    qubo = Qubo(s)
    qubo.calculate_qubo()
    pickle.dump(qubo, open(f"./src/logs/{name}.p", "wb"))


def solve_qubo(file, samples=1000):

    print("simulation")
    run_simulation(file, samples)

    print("dwave")
    run_uqo_dwave(file, samples)

    print("DAU")
    run_dau(file, samples)


def run_dau(file, samples):
    qubo = pickle.load(open(f"./src/logs/{file}", "rb"))

    qubo_dau = deepcopy(qubo)
    map1 = qubo_dau.s_to_nr_mapping
    map2 = qubo_dau.nr_to_s_mapping

    qubo_matrix = {(map1[s1], map1[s2]): v for (
        s1, s2), v in qubo_dau.J.items()}

    dau_sol = solve_dau(qubo_matrix, samples)
    *_, response = dau_sol
    r = {map2[s]: v for s, v in response.samples()[0].items()}
    qubo_dau.interpret_solution_dict(r)
    pickle.dump(qubo_dau, open(f"./src/logs/dau_qubo{file}", "wb"))
    pickle.dump(dau_sol, open(f"./src/logs/dau_response_{file}", "wb"))


def run_uqo_dwave(file, samples):
    qubo = pickle.load(open(f"./src/logs/{file}", "rb"))
    qubo_dw = deepcopy(qubo)
    map1 = qubo_dw.s_to_nr_mapping
    map2 = qubo_dw.nr_to_s_mapping
    qubo_matrix = {(map1[s1], map1[s2]): v for (
        s1, s2), v in qubo_dw.J.items()}
    dw_sol = solve_dwave(qubo_matrix, samples)
    *_, response = dw_sol
    r = {map2[s]: v for s, v in response.samples()[0].items()}
    qubo_dw.interpret_solution_dict(r)
    pickle.dump(qubo_dw, open(
        f"./src/logs/dwave_qubo{file}", "wb"))
    pickle.dump(dw_sol, open(f"./src/logs/dwave_response_{file}", "wb"))


def run_dwave(file, samples):
    qubo = pickle.load(open(f"./src/logs/{file}", "rb"))
    qubo_dw = deepcopy(qubo)
    sampler = DWaveSampler(solver=dict(qpu=True))
    sampler = EmbeddingComposite(sampler)
    answer = sampler.sample_qubo(
        {x: y for x, y in qubo_dw.J.items() if y != 0}, num_reads=samples)
    qubo_dw.interpret_solution_dict(
        {x: y for x, y in answer.samples()[0].items()})
    pickle.dump(qubo_dw, open(
        f"./src/logs/dwave_direct_qubo{file}", "wb"))
    pickle.dump(answer, open(f"./src/logs/dwave_direct_response{file}", "wb"))


def run_simulation(file, samples):
    qubo = pickle.load(open(f"./src/logs/{file}", "rb"))
    qubo_sim = deepcopy(qubo)
    qubo_matrix = qubo_sim.J

    sim_sol = solve_simulated_annealing(qubo_matrix, samples)
    response = sim_sol
    qubo_sim.interpret_solution_dict(response.samples()[0])
    pickle.dump(qubo_sim, open(f"./src/logs/sim_qubo{file}", "wb"))
    pickle.dump(sim_sol, open(f"./src/logs/sim_response_{file}", "wb"))


def run_scaling_experiment(s1, plot_result=False, path="./logs/scaling_logs", samples=100):
    qubo1 = Qubo(s1)
    qubo1.penalty_terms[5] = 0
    qubo1.calculate_qubo()
    print(f"qbits: {len(qubo1.h)}")
    answer1 = solvelog_dwave(qubo1, samples)
    ts = int(datetime.datetime.now().timestamp()*1000)
    name = f"{len(s1.workload)}-{s1.machine_park.nr_m}-{s1.time_max}_{ts}"
    df_answer = answer1.to_pandas_dataframe()
    pickle.dump(df_answer, open(
        f"{path}/dwave_1_{name}.p", "wb"))
    if plot_result:
        *temp2, _ = qubo1.interpret_solution_dict(
            {x: y for x, y in answer1.samples()[0].items()})
        print(temp2)
        qubo1.plot_solution()
    return qubo1


def load_experiments(s1, path="./logs/scaling_logs"):
    name = f"dwave_1_{len(s1.workload)}-{s1.machine_park.nr_m}-{s1.time_max}"
    df_lst = [pickle.load(open(join(path, f), "rb"))
              for f in listdir(path) if (isfile(join(path, f)) and f.startswith(name))]
    return df_lst


if __name__ == "__main__":
    from os import walk
    for *_, q_file in walk("./src/logs"):
        for q in [x for x in q_file if x.startswith("qubo")]:
            # run_simulation(q, 100)
            # run_dwave(q, 100)

            # not tested since uqo does not work
            # run_uqo_dwave(q, 100)
            # run_dau(q, 100)
            pass
