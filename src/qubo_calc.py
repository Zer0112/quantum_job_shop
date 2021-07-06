def calculate_qubo_H1_h(j, m, t, times, vm=0, vj=0):
    return 1-2*times


def calculate_qubo_H1_J(j1, m1, t1, j2, m2, t2, vm=0, vj=0):
    if j1 == j2 and m1 == m2:
        return 2
    else:
        return 0


def calculate_qubo_H2_h(j, m, t,):
    return 0


def calculate_qubo_H2_J(j1, m1, t1, j2, m2, t2, t_mj_operation, vm1=-1, vm2=-1, vj1=-1, vj2=-1):
    if m1 == m2 and abs(t1 - t2) < t_mj_operation and j2 == j1:

        if vj1 == vj2 and vm1 == vm2:
            return 1
    if m1 == m2 and abs(t1 - t2) < t_mj_operation and vm1 == vm2:
        # ugly test to see if we working with virtuell states
        # it is needed for the upper diagonal matrix
        if vj1 == -1:
            return 2
        else:
            return 4
    else:
        return 0


def calculate_qubo_H3_h(j, m, t, b_j=1, vm=0, vj=0):
    return 0


def calculate_qubo_H3_J(j1, m1, t1, j2, m2, t2, t_mj_operation, vm1=-1, vm2=-1, vj1=-1, vj2=-1):

    if m1 == m2 and abs(t1 - t2) < t_mj_operation and j2 == j1:
        if vj1 == vj2 and vm1 == vm2:
            return 1
    if j1 == j2 and abs(t1 - t2) < t_mj_operation and vm1 == vm2:
        if vj1 == -1:
            return 2
        else:
            return 4
    else:
        return 0


def calculate_qubo_H4_h(j, m, t):
    if m == 0 or j == 0:
        return 0
    # TODO
    return 0


def calculate_qubo_H4_J(j1, m1, t1, j2, m2, t2):
    if m1 == 0 or m2 == 0 or j1 == 0 or j2 == 0:
        return 0
    # TODO
    return 0


def calculate_qubo_H5_h(j, m, t, deadline, tmax):
    if deadline == 0:
        return 0.2*(tmax-t)/tmax
    # TODO
    return 0.2*(deadline-t)/tmax


def calculate_qubo_H5_J(j1, m1, t1, j2, m2, t2):

    # TODO
    return 0


def calculate_qubo_H6_h(j, m, t, t0_j):
    if not m == 0:
        return 0
    return 0


def calculate_qubo_H6_J(j1, m1, t1, j2, m2, t2, d_j):
    if j1 == j2 and m1 == m2 and t1 > t2 and m1 == 0 and t1 < d_j:
        return 0
    else:
        return 0
