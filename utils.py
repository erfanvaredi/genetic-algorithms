import pandas as pd
import numpy as np


def calc_loan_revenue(df_customers: pd.DataFrame, bank_expected_loan_loss: float, bank_loan_interest_rate: float):
    """
    loan revenue as v
    :param df_customers: dataset of customers
    :param bank_expected_loan_loss: of target bank
    :param bank_loan_interest_rate: of target bank rL
    :return:
    """
    v = 0
    for row_index in range(len(df_customers)):
        row = df_customers.iloc[row_index]
        v += (row['Loan Size'] * bank_loan_interest_rate) - bank_expected_loan_loss
    return v


def calc_loan_cost(df_customers: pd.DataFrame, bank_predetermined_institutional_cost):
    """
    loa cost as mue
    :param df_customers: dataset of customers
    :param bank_predetermined_institutional_cost: gama
    :return:
    """
    mue = 0
    for row_index in range(len(df_customers)):
        row = df_customers.iloc[row_index]
        mue += (row['Loan Size'] * bank_predetermined_institutional_cost)
    return mue


def calc_total_transaction_cost(df_customers: pd.DataFrame, bank_loan_interest_rate: float, bank_required_reserve_ratio,
                                financial_institutions_deposit):
    """
    as w_bar
    :param df_customers:
    :param bank_loan_interest_rate: rL
    :param bank_required_reserve_ratio: K
    :param financial_institutions_deposit: D
    :return:
    """
    w_bar = 0
    for row_index in range(len(df_customers)):
        row = df_customers.iloc[row_index]
        K = bank_required_reserve_ratio
        L = row['Loan Size']
        D = financial_institutions_deposit
        T = (1 - K) * D - L
        w_bar += bank_loan_interest_rate * T
    return w_bar


def calc_cost_of_demand_deposit(rD, financial_institutions_deposit):
    """
    as beta
    :param rD: weighted average of all different deposits rates based on deposit type (checking, saving) and based on
                deposit age ranging from (3 months till 10 years)
    :param financial_institutions_deposit: as D
    :return:
    """
    return rD * financial_institutions_deposit


def calc_sum_of_landa(df_customers: pd.DataFrame()):
    som_of_landa = 0
    for row_index in range(len(df_customers)):
        row = df_customers.iloc[row_index]
        som_of_landa += (get_creadit_landa_from_rating(row['Credit Rating']))
    return som_of_landa


def get_creadit_landa_from_rating(credit_rating: str):
    credit_rating = credit_rating.replace(' ', '')
    if credit_rating == 'AAA':
        return np.random.uniform(0.0002, 0.0003)
    elif credit_rating == 'AA':
        return np.random.uniform(0.0003, 0.001)
    elif credit_rating == 'A':
        return np.random.uniform(0.001, 0.0024)
    elif credit_rating == 'BBB':
        return np.random.uniform(0.0024, 0.0058)
    elif credit_rating == 'BB':
        return np.random.uniform(0.0058, 0.0119)
    else:
        return np.nan


def calc_fitness(df_customers: pd.DataFrame(), bank_expected_loan_loss, bank_loan_interest_rate,
                 bank_required_reserved_ratio, financial_institutions_deposit, rD):
    v = calc_loan_revenue(df_customers, bank_expected_loan_loss, bank_loan_interest_rate)
    w_bar = calc_total_transaction_cost(df_customers, bank_loan_interest_rate, bank_required_reserved_ratio,
                                        financial_institutions_deposit)
    beta = calc_cost_of_demand_deposit(rD, financial_institutions_deposit)
    sum_of_landa = calc_sum_of_landa(df_customers)

    return v + w_bar - beta - sum_of_landa


def is_GAMCC_satisfied(df_customers: pd.DataFrame, bank_required_reserve_ratio, financial_institutions_deposit) -> bool:
    """
    check if sum of loan amounts are less than (1-K)D
    :param df_customers:
    :return:
    """
    sum_of_loan_amount = 0
    K = bank_required_reserve_ratio
    D = financial_institutions_deposit

    for row_index in range(len(df_customers)):
        row = df_customers.iloc[row_index]
        L = row['Loan Size']
        sum_of_loan_amount += L

    return sum_of_loan_amount <= ((1 - K) * D)


def get_loan_category(loan):
    if loan <= 13000:
        return 'micro'
    elif 13000 < loan <= 50000:
        return 'small'
    elif 50000 < loan <= 100000:
        return 'medium'
    elif 100000 < loan <= 250000:
        return 'large'
    else:
        return np.nan


def get_landa_category(landa):
    if 0.0002 <= landa <= 0.0003:
        return 'AAA'
    elif 0.0003 < landa <= 0.001:
        return 'AA'
    elif 0.001 < landa <= 0.0024:
        return 'A'
    elif 0.0024 < landa <= 0.0058:
        return 'BBB'
    elif 0.0058 < landa <= 0.0119:
        return 'BB'
    else:
        return np.nan


def get_loan_age_category(loan_age):
    if 1 <= loan_age <= 3:
        return 1
    elif 3 < loan_age <= 5:
        return 2
    elif 5 < loan_age <= 10:
        return 3
    elif 10 < loan_age <= 20:
        return 4
    else:
        return np.nan


def init_population(customer_size: int, population_size: int) -> np.ndarray:
    chromos = []
    for i in range(population_size):

        arr = np.zeros(customer_size, dtype=bool)

        for i in range(len(arr)):
            rand = np.random.randint(1, 100)
            if rand > 50:
                np.put(arr, [i], [True])

        chromos.append(arr)

    return np.array(chromos)


def init_population_with_GAMCC(df_customers: pd.DataFrame, customer_size: int, population_size: int,
                               bank_required_reserve_ratio, financial_institutions_deposit) -> np.ndarray:
    chromos = []
    while len(chromos) <= population_size:

        chromo = np.zeros(customer_size, dtype=bool)

        arr_loan_candidate = []
        for i in range(len(chromo)):
            rand = np.random.randint(1, 100)
            if rand > 50:
                np.put(chromo, [i], [True])
                arr_loan_candidate.append(i)

        # check GAMCC
        this_chromo_df = df_customers.iloc[arr_loan_candidate, :]
        if is_GAMCC_satisfied(this_chromo_df, bank_required_reserve_ratio, financial_institutions_deposit):
            chromos.append(chromo)
            print('chromo_accepted_count: ', len(chromos))
            # else:
            # print('chromo: ', chromo, ' rejected')

    return np.array(chromos)


def calc_all_fitness_of_chromos(df_customers: pd.DataFrame, chromos: np.ndarray, rD, bank_required_reserve_ratio,
                                financial_institutions_deposit, bank_loan_interest_rate) -> np.ndarray:
    """

    :param chromos:
    :param rD -> for senario 1 is 0.009
    :return: vector of chromos fitness
    """
    # bank_required_reserve_ratio = 0
    # financial_institutions_deposit = 5
    # bank_loan_interest_rate = 0.01

    chromos_fitness_vector = np.zeros(chromos.shape[0])

    for row in range(chromos.shape[0]):
        sum_of_fitness_of_chromo = 0
        # persons who are candidate to get loan
        arr_loan_candidate = []
        for gene_index in range(len(chromos[row,])):
            if chromos[row, gene_index]:
                arr_loan_candidate.append(gene_index)

        fitness = calc_fitness(df_customers=df_customers.iloc[arr_loan_candidate, :], bank_expected_loan_loss=0.001,
                               bank_loan_interest_rate=bank_loan_interest_rate,
                               bank_required_reserved_ratio=bank_required_reserve_ratio,
                               financial_institutions_deposit=financial_institutions_deposit, rD=rD)
        np.put(chromos_fitness_vector, [row], [fitness])

    return chromos_fitness_vector

    # def get_candidate_customers_from_chromo(chromo)


def get_rated_fit_vector(chromos_fitness_vector: np.ndarray) -> np.ndarray:
    sum_of_all_fit = chromos_fitness_vector.sum()
    func_rated_fit = np.vectorize(lambda x: x / sum_of_all_fit)
    return func_rated_fit(chromos_fitness_vector)


def xover(parent1: np.ndarray, parent2: np.ndarray):
    rand_spliter = np.random.randint(0, len(parent1))
    child1 = parent1
    child2 = parent2
    np.put(child1, list(range(rand_spliter, len(parent1))), parent2[rand_spliter:])
    np.put(child2, list(range(0, rand_spliter)), parent1[:rand_spliter])
    return child1, child2


def mutate(parent1: np.ndarray, parent2: np.ndarray):
    rand_pos_for_mutate = np.random.randint(0, len(parent1))
    child1 = parent1
    child2 = parent2
    ch1_gene = True
    ch2_gene = True
    if parent2[rand_pos_for_mutate]:
        ch1_gene = False
    if parent1[rand_pos_for_mutate]:
        ch2_gene = False
    np.put(child1, [rand_pos_for_mutate], [ch1_gene])
    np.put(child2, [rand_pos_for_mutate], [ch2_gene])
    return child1, child2


def delete_chromo_based_on_bad_fit(chromos: np.ndarray, fitness_vector,
                                   number_of_chromo_to_be_deleted: int) -> np.ndarray:
    selected_chromos = []
    while len(selected_chromos) < number_of_chromo_to_be_deleted:
        min_checked_val = fitness_vector.max()
        min_found_index = 0
        for chromo_index in range(len(fitness_vector)):
            if chromo_index in selected_chromos:
                continue
            if min_checked_val > fitness_vector[chromo_index]:
                min_checked_val = fitness_vector[chromo_index]
                min_found_index = chromo_index
        selected_chromos.append(min_found_index)

    target_chromos_index = [item for item in list(range(0, len(chromos))) if item not in selected_chromos]

    return chromos[target_chromos_index, :]


def get_dataframe_by_chromo(df_customers: pd.DataFrame, chromo) -> pd.DataFrame:
    selected_customers = []
    for i in range(len(chromo)):
        if chromo[i]:
            selected_customers.append(i)

    return df_customers.iloc[selected_customers, :]
