import numpy as np
import torch 
from itertools import combinations_with_replacement 
import pandas as pd

class Cluster():
    def __init__(self, model, start_sigma = 1, end_sigma=5, num_rounds = 10, zero_fix = False, inc=1, b = 7, sigma_join = 'mean_mu'):
        self.current_n_clusters = 1
        self.model = model
        self.min_sigma = start_sigma
        self.max_sigma = end_sigma
        self.num_rounds = num_rounds
        self.b = b
        self.sigmas = np.linspace(start_sigma, end_sigma, num_rounds)
        self.how_many_sigma = self.sigmas[0]
        self.zero_fix= zero_fix
        self.sigmas_join = sigma_join
        self.inc = inc
    
    def grab_all_model_weights(self):
        weights = {}       
        weights['mu'] = np.array([])
        weights['sigma'] = np.array([])
        for i, (n, p) in enumerate(self.model.named_parameters()):
            if 'mu' in n and 'prior' not in n:
                weights['mu'] = np.append(weights['mu'], torch.flatten(p).detach())
            elif 'sigma' in n:
                weights['sigma'] = np.append(weights['sigma'], torch.flatten(p).detach())
        weights['mu'] = np.array(weights['mu'])
        weights['sigma'] = np.array(weights['sigma'])
        # set all sigmas to be a minimum of 2**-20
        weights['sigma'][weights['sigma'] < 2**-20] = 2**-20
        return weights

    def assign_sigmas(self, assigned):
        for n, p in self.model.named_parameters():
            if 'sigma' in n and 'prior' not in n:
                number = p.numel()
                assignment = torch.Tensor(assigned[:number]).reshape(p.shape)
                is_fixed = ~torch.isinf(torch.Tensor(assigned[:number]).reshape(p.shape))
                assignment[torch.isinf(assignment)] = p[torch.isinf(assignment)]
                p.data = assignment
                if not self.sigmas_join == 'allow_retraining':
                    p.is_fixed = torch.where(is_fixed)
                    p.not_fixed = torch.where(~is_fixed)
                assigned = assigned[number:]
                self.model.state_dict()[n] = p.data

    def assign_weights(self, assigned):
        for n, p in self.model.named_parameters():
                if 'mu' in n and 'prior' not in n:
                        number = p.numel()
                        assignment = torch.Tensor(assigned[:number]).reshape(p.shape)
                        is_fixed = ~torch.isinf(torch.Tensor(assigned[:number]).reshape(p.shape))
                        assignment[torch.isinf(assignment)] = p[torch.isinf(assignment)]
                        p.data = assignment
                        p.is_fixed = torch.where(is_fixed)
                        p.not_fixed = torch.where(~is_fixed)
                        assigned = assigned[number:]
                        self.model.state_dict()[n] = p.data

    def create_possible_centroids(self, powl, maximum_weight):
           max_pow_2 = int(np.ceil(np.log2(maximum_weight))) 
           powers_of_2 = np.append(2.0**np.arange(-self.b, max_pow_2, 1), [0])
           cb = combinations_with_replacement(powers_of_2, powl)
           possible_centroids = np.array(list(map(np.sum, list(combinations_with_replacement(powers_of_2, powl)))))
           possible_centroids = possible_centroids[np.abs(possible_centroids) <= maximum_weight]
           pw = np.unique(possible_centroids)
           return np.unique(np.concatenate([np.flip(-pw, [0]), pw]))


    def get_summary(self,cluster, data, sigmas):
        if not np.isnan(data).any() and len(data) >= 1:
            sigmas = np.nan_to_num(sigmas)
            return [cluster, np.max(data), np.min(data), np.std(data), np.mean(data), np.median(data), np.max(sigmas), np.min(sigmas), np.std(sigmas), np.mean(sigmas), np.median(sigmas)]
        else:
            return []

    def find_the_next_cluster(self, weights, sigmas, clusters, assignment_mu, assignment_sigma, left_to_assign):
        not_assigned_indicies = np.where(assignment_mu == np.inf)[0]
#        print(clusters[zero_i], 'is zero')
        distances = np.ones((clusters.shape[0], weights.shape[0]))*10000
        for i, w in enumerate(clusters):
            distances[i, :] = np.abs(np.subtract(w , weights)) / (sigmas)
        if self.zero_fix:
         zero_i = clusters.shape[0]//2
         to_zero = np.abs(weights) < 2**-self.b
         try:
            distances[zero_i, to_zero] = 0.0
         except: 
            print('zero fix failed')

        small_enough = np.zeros_like(distances)
        small_enough[distances <= self.how_many_sigmas] = True
        count = small_enough.sum(axis=1)
        try:
            max_count_cluster = np.argmax(count)
            local_cluster_distances = distances[max_count_cluster, :]
            sorted_indexes = np.argsort(local_cluster_distances) #sort them by index
            if local_cluster_distances[sorted_indexes[0]] <= self.how_many_sigmas:
                number_assigned = np.max(np.where(local_cluster_distances[sorted_indexes] <= self.how_many_sigmas)) + 1
                if left_to_assign <= number_assigned:
                    number_assigned = left_to_assign
            else:
                number_assigned = 0
        except Exception as e:
            number_assigned = 0
        if number_assigned == 0:
            return assignment_mu, assignment_sigma, 0, 0, left_to_assign, 0
        indicies_of_not_assigned_to_be_allocated = sorted_indexes[:number_assigned]
        to_be_assigned = not_assigned_indicies[indicies_of_not_assigned_to_be_allocated]
        assignment_mu[to_be_assigned] = clusters[max_count_cluster]
        #assignment_sigma[to_be_assigned] = np.std(weights[indicies_of_not_assigned_to_be_allocated])
        assignment_sigma[to_be_assigned] = self.sigma_join_fn(weights[indicies_of_not_assigned_to_be_allocated], sigmas[indicies_of_not_assigned_to_be_allocated])
        #assignment_sigma[to_be_assigned] = 0
        return assignment_mu, assignment_sigma, clusters[max_count_cluster], number_assigned, to_be_assigned, self.get_summary(clusters[max_count_cluster], weights[indicies_of_not_assigned_to_be_allocated], sigmas[indicies_of_not_assigned_to_be_allocated]) 

    def sigma_join_fn(self, mus, sigmas):
        if self.sigmas_join == 'mean_sigma':
            # sqrt of the variance 
            return 1/np.sqrt(np.mean(sigmas**2))
        if self.sigmas_join == 'std_mu':
            return np.std(mus)
        if self.sigmas_join == 'keep_the_same_divide_by_2':
            return sigmas / 2
        if self.sigmas_join == 'keep_the_same_divide_by_10':
            return sigmas / 10
        if self.sigmas_join == 'allow_retraining':
            return sigmas


    def cluster_round(self, percentage_to_be_clustered, round):
        number = 0
        weights = self.grab_all_model_weights()
        total_num = weights['mu'].shape[0]
        powl = 1 
        counter = 1
        assignment_mu = np.full_like(weights['mu'], np.inf)
        assignment_sigma = np.full_like(weights['sigma'], np.inf)
        assigned_clusters = []
        self.how_many_sigmas = self.sigmas[round]
        left_to_assign = int(np.ceil(total_num * percentage_to_be_clustered))
        round_summary = pd.DataFrame()
        while left_to_assign > 0:
            not_assigned = weights['mu'][np.isinf(assignment_mu)]
            not_assigned_sig =  weights['sigma'][np.isinf(assignment_mu)]
            max_not_assigned = np.max(np.abs(not_assigned))
            if np.isinf(max_not_assigned) or max_not_assigned == 0.0:
                print(not_assigned, 'problem here')

            clusters = self.create_possible_centroids(powl, max_not_assigned)
            still_assigning = True
            while still_assigning:
                assignment_mu, assignment_sigma, assigned_cluster, number_assigned, assigned_indicies, summary = self.find_the_next_cluster(not_assigned, not_assigned_sig, clusters, assignment_mu, assignment_sigma, left_to_assign)
                if number_assigned < 1:
                    still_assigning = False
                    continue 
                summary.extend([number_assigned, percentage_to_be_clustered, round])
                assigned_clusters.append(assigned_cluster)
                not_assigned = weights['mu'][np.isinf(assignment_mu)]
                not_assigned_sig = weights['sigma'][np.isinf(assignment_sigma)]
                left_to_assign -= number_assigned
                clusters = np.delete(clusters, np.where(clusters == assigned_cluster))
                keys = ['value', 'w_max_prev', 'w_min_prev', 'w_std_prev', 'w_mean_prev', 'w_median_prev', 's_max_prev', 's_min_prev', 's_std_prev', 's_mean_prev', 's_median_prev', 'number_assigned', 'percent', 'round']
                output_result = pd.DataFrame([dict(zip(keys, summary))])
                round_summary = pd.concat([round_summary, output_result])
            if left_to_assign < 1:
                continue
            if powl < 4 and counter % 2 == 1:
                powl += 1
                counter += 1
                print('increasing powl to ', powl)
            else:
                counter += 1
                self.how_many_sigmas *= self.inc
                print('increasing how many sigmas to ', self.how_many_sigmas)
        self.assign_weights(assigned=assignment_mu)
        self.assign_sigmas(assigned=assignment_sigma)
        not_assigned = weights['mu'][np.isinf(assignment_mu)]
        assigned_clusters = np.unique(assigned_clusters)
#        print(f'The clusters are {assigned_clusters} and we have assigned {number} which is {number/total_num}')
        return assigned_clusters, number/total_num, not_assigned, round_summary



