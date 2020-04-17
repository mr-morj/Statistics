import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from statsmodels.stats.weightstats import _tconfint_generic
from math import sqrt

#стартовые значения
#выбираем сами для разных моделей
P_inf = 0.4 #вероятность того что зараженный заразит здорового
P_die = 0.04 #вероятность зараженного умереть

N = 10
n_days = 90

#N = int(input('Enter the number of family members: '))
#P_inf = float(input('Enter the probability of infecting someone: '))
#P_die = float(input('Enter the probability of dying from the virus: '))
#считаем латентный период до 7 дней
#считаем активный период до 14 дней

#вероятности берем для проверки работоспособности
P_latent = [0.01, 0.03, 0.05, 0.2, 0.6, 0.1, 0.01]
P_active = [0.005, 0.02, 0.04, 0.015, 0.02, 0.035, 0.15, 0.5, 0.09, 0.04, 0.04, 0.02, 0.02, 0.005]

x_latent = [x+1 for x in range(len(P_latent))]
x_active = [x+1 for x in range(len(P_active))]

latent_series = pd.DataFrame({'Day': x_latent, 'Prob': P_latent}).set_index('Day')
active_series = pd.DataFrame({'Day': x_active, 'Prob': P_active}).set_index('Day')

#просто сохраним
latent_series.to_csv('Latent Series.csv')
active_series.to_csv('Active Series.csv')

def stats_graph(n_a, n_he, n_l, n_de, n_re, name):
    plt.grid(alpha=0.4)
    plt.plot(n_a, c='r', label='Active inf case')    
    plt.plot(n_he, c='g', label='Healthy case')
    plt.plot(n_l, c='y', label='Latention case')
    plt.plot(n_de, c='k', label='Die case')
    plt.plot(n_re, c='c', label='Recover case')
    plt.title(name)
    plt.xlabel('Days after first infection')
    plt.ylabel('Number of members')
    plt.legend(loc = 'best')
    plt.show()

def simulation(number):
    #создаем таблицу информации по дням для каждого члена семьи
    family_model = pd.DataFrame({'Day': [x for x in range(n_days+1)]}).set_index('Day')
    
    for n in range(N):
        family_model[f'N{n+1}'] = 'he'
    
    #he := здоровый (не болел)
    #de := умер
    #li := латентный на i дне
    #ai := активно заражающий на i дне
    #re := выздоровел и получил имунитет
    
    #инициализация первого члена семьи в латентный период
    start_chlen = np.random.randint(N)
    all_family = list(family_model.columns)
    
    family_model.iloc[0, start_chlen] = 'l0'
    
    #сопутствующие счетчики и массиві для каждого типа членов семьи
    n_he = N-1
    n_he_array = [n_he]
    
    all_h = all_family.copy()
    all_h.remove(family_model.columns.values[start_chlen])
    
    n_de = n_a = n_re = 0
    all_de = list()
    all_a = list()
    all_re = list()
    
    n_re_array = [0]
    n_de_array = [0] 
    n_a_array = [0]
    
    
    n_l = 1
    n_l_array = [n_l]
    all_l = list()
    all_l.append(family_model.columns.values[start_chlen])
    
    #моделирование єксперимента
    for day in range(0, n_days, 1):
        #print(f'-----------Day {day}-----------')
        for user in range(N):
            cur = family_model.iloc[day, user]
            
            #латентные
            if re.findall(r'\w', cur)[0]=='l':
                id = int(re.findall(r'\d+', cur)[0])
                check_nextday = np.random.rand()
                if id!=len(P_latent):
                    #каждій день новій рандом и проверка
                    #если рандом больше, чем значение в ряде, то на следующий день латентная фаза продолжается
                    if check_nextday >= latent_series.Prob.iloc[id]:
                        #print(f'Next day +1 latent mood! check {check_nextday}')
                        family_model.iloc[day+1, user] = f'l{id+1}'
                    #в противном случае переходит в активную фазу
                    else:
                        #print(f'{all_family[user]} has a active stage! (Check prob {check_nextday})')
                        family_model.iloc[day+1, user] = 'a1'
                        n_l = n_l - 1
                        all_l.remove(family_model.columns.values[user])
                        n_a = n_a + 1
                        all_a.append(family_model.columns.values[user])
                #если все дни прошли, то сами определяем что человек переходит в активную фазу        
                else:
                    #print(f'{all_family[user]} has a active stage! (End 7 Latent period)')
                    family_model.iloc[day+1, user] = 'a1'
                    n_l = n_l - 1
                    all_l.remove(family_model.columns.values[user])
                    n_a = n_a + 1
                    all_a.append(family_model.columns.values[user])
            
            #активные
            if re.findall(r'\w', cur)[0]=='a':
                id = int(re.findall(r'\d+', cur)[0])
                check_nextday = np.random.rand()
                check_die = np.random.rand()
                
                #каждій день новій рандом и проверка
                #если рандом больше, чем значение в ряде, то на следующий день активная фаза продолжается
                if id!=len(P_active):
                    if check_nextday >= active_series.Prob.iloc[id]:
                        #print(f'Next day +1 active mood! check {check_nextday}')
                        family_model.iloc[day+1, user] = f'a{id+1}'
                    #в противном случае или умирает или выздоровел
                    else:
                        if check_die < P_die:
                            #print(f'{all_family[user]} has died! (Check prob {check_die})')
                            family_model.iloc[day+1:, user] = 'de'
                            n_a = n_a - 1
                            all_a.remove(family_model.columns.values[user])
                            n_de = n_de + 1
                            all_de.append(family_model.columns.values[user])
                        else:
                            #print(f'{all_family[user]} has recovered! (Check prob {check_die})')
                            family_model.iloc[day+1:, user] = 're'
                            n_a = n_a - 1
                            all_a.remove(family_model.columns.values[user])
                            n_re = n_re + 1
                            all_re.append(family_model.columns.values[user])
                #прошел период активной фазі, сами определяем и рандомим умрет или віздоров        
                else:
                    if check_die < P_die:
                        #print(f'{all_family[user]} has died! (End 14 active period)')
                        family_model.iloc[day+1:, user] = 'de'
                        n_a = n_a - 1
                        all_a.remove(family_model.columns.values[user])
                        n_de = n_de + 1
                        all_de.append(family_model.columns.values[user])
                    else:
                        #print(f'{all_family[user]} has recovered! (End 14 active period)')
                        family_model.iloc[day+1:, user] = 're'
                        n_a = n_a - 1
                        all_a.remove(family_model.columns.values[user])
                        n_re = n_re + 1
                        all_re.append(family_model.columns.values[user])
                        
                        
                #генерация новых заражений
                check_inf = np.random.rand()
                #проверка можно ли кого-то заразить
                if len(all_h)!=0: 
                    #есил меньше дефолтной вероятности, то заражаем кого-то
                    if check_inf < (1-(1-P_inf)**(n_a)):
                        #рандомим кого заразить
                        select_victim = np.random.randint(n_he)
                        victim = all_h[select_victim]
                        #print(f'New victim {victim}! (Check prob {check_inf})')
                        family_model[victim].iloc[day+1]='l1'
                        all_l.append(victim)
                        n_l = n_l + 1
                        all_h.remove(victim)
                        n_he = n_he - 1
        #обновляем значения массивов в конце каждого дня          
        n_l_array.append(n_l)
        n_he_array.append(n_he)
        n_de_array.append(n_de)
        n_re_array.append(n_re)
        n_a_array.append(n_a)
                
    #stats_graph(n_a_array, n_he_array, n_l_array, n_de_array, n_re_array, 'Family Stats ' +str(number) +' simulation')        
    return n_a_array, n_he_array, n_l_array, n_de_array, n_re_array

def stats_array_by_case(array, days, n, case):
    mean_case_by_days = []
    std_case_by_days = []
    for day in range(days):
        array_day = []
        for i in range(n):
            array_day.append(array[i][case][day])
        array_day = np.array(array_day)
        mean_case_by_days.append(array_day.mean())
        std_case_by_days.append(array_day.std(ddof=1)/sqrt(len(array_day)))
        
    return np.array(mean_case_by_days), np.array(std_case_by_days)


def MC_simulation(times):
    sim = []
    for id in range(times):
        #print(f'\n-- {id+1} SIMULATION -- {id+1} SIMULATION -- {id+1} SIMULATION --\n')
        sim.append(simulation(id+1))
    
    mean_a_days, std_a_days = stats_array_by_case(sim, n_days, times, 0)
    mean_he_days, std_he_days = stats_array_by_case(sim, n_days, times, 1)
    mean_l_days, std_l_days = stats_array_by_case(sim, n_days, times, 2)
    mean_de_days, std_de_days = stats_array_by_case(sim, n_days, times, 3)
    mean_re_days, std_re_days = stats_array_by_case(sim, n_days, times, 4)
    
    stats_graph(mean_a_days, mean_he_days, mean_l_days, mean_de_days, mean_re_days, 'Monte-Carlo results after '+str(times)+' simulation')
    
    #оценки при помощи т-интервала (мы не знаем дисперсию)
    day_check = int(input('Enter the day number for which you want to see statistics: '))
    
    print(f'\nActual Active case mean in {day_check} day: ', mean_a_days[day_check])
    print(f'Active case mean 95% confidence interval in {day_check} day: ', _tconfint_generic(mean_a_days[day_check], std_a_days[day_check], times-1, 0.05, 'two-sided'))
    
    print(f'\nActual Heath case mean in {day_check} day: ', mean_he_days[day_check])
    print(f'Heath case mean 95% confidence interval in {day_check} day: ', _tconfint_generic(mean_he_days[day_check], std_he_days[day_check], times-1, 0.05, 'two-sided'))
    
    print(f'\nActual Latent case mean in {day_check} day: ', mean_l_days[day_check])
    print(f'Latent case mean 95% confidence interval in {day_check} day: ', _tconfint_generic(mean_l_days[day_check], std_l_days[day_check], times-1, 0.05, 'two-sided'))
    
    print(f'\nActual Die case mean in {day_check} day: ', mean_de_days[day_check])
    print(f'Die case mean 95% confidence interval in {day_check} day: ', _tconfint_generic(mean_de_days[day_check], std_de_days[day_check], times-1, 0.05, 'two-sided'))
    
    print(f'\nActual Recover case mean in {day_check} day: ', mean_re_days[day_check])
    print(f'Recover case mean 95% confidence interval in {day_check} day: ', _tconfint_generic(mean_re_days[day_check], std_re_days[day_check], times-1, 0.05, 'two-sided'))
    

MC_simulation(1000)
        
            
        

