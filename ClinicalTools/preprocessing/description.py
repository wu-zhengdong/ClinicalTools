# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from scipy.stats import levene

import warnings
warnings.filterwarnings('ignore')


class Aid:
    def __init__(self, label, dataframe, class_threshold = 10, multicate_var = None):
        """
        Created on Fri Jan 22 09:13:53 2021
        @author: MZH

        Example:
            import aidcloudroutine.Analysis as ana
            aid = ana.Aid(label = 'label', dataframe = full, class_threshold = 10, multicate_var = None)
            full_stat = aid.stats_table(full_data, group0, group1)
            --------------------------------------------------------------------------------
            No variable is valid for parametric test
            
                            All(N=#)          Group 0(n=#)         Group 1(n=#)    Statistics     P_value
            年龄            52.65(14.55)    51.74(14.49)           54.42(14.57)                   0.202
            体重指数         22.98(3.91)     23.23(4.09)           22.49(3.52)                     0.138
            --------------------------------------------------------------------------------
            ana.Aid.data_info(data)
            --------------------------------------------------------------------------------
            	第1次-年龄(年)	      性别	    住院天数	
                Dtypes	int64	     int64	    float64	
                Num_Unique_Value	   37	      2	
                Missing_Count	       0	      0	
                Missing_Percentage	   0	      0	
            --------------------------------------------------------------------------------
        """
        self.label = label
        num_class = dataframe.nunique()
        if multicate_var == None:
            self.multicate_var = list(num_class.index[((num_class>2)&(num_class<=class_threshold))])
        else:
            self.multicate_var = multicate_var
        binary_var = list(num_class.index[num_class<=2])
        binary_var.remove(label)
        self.binary_var = [i for i in binary_var if i not in self.multicate_var]
        self.OOV = list(num_class.index[num_class == 1])
        self.cate_var = self.binary_var + self.multicate_var
        self.cont_var = [i for i in dataframe.columns if i not in self.cate_var and i != self.label]
        self.class_threshold = class_threshold
        print('Categorical variables: {} ({} binary variables ,{} multi-category variables);\
              \nContinuous variables: {};\nOOV variables: {}\nTarget Label: {};'.format(len(self.cate_var),len(self.binary_var),len(self.multicate_var),
              len(self.cont_var),self.OOV,self.label))

    def data_info(self, df):
        """
        Input: Dataframe
        Output: Dataframe with 4 rows: 1: Datatype of each feature; 2: # of unique value for each feature 
        3:# of Missing for each feature;4:Missing Percentage for each feature
        """
        info_tab = pd.DataFrame({'Dtypes':df.dtypes}).T
        info_tab = info_tab.append(pd.DataFrame({'Num_Unique_Value':df.apply(lambda x: len(x.unique()))}).T)
        info_tab= info_tab.append(pd.DataFrame({'Missing_Count':np.sum(df.isna())}).T)
        info_tab = info_tab.append(pd.DataFrame({'Missing_Percentage':(np.sum(df.isna()))/(df.shape[0])}).T)
        return(info_tab)
    
    def Norm_var_test(self, group0, group1, alpha = 0.05):
        """
        Carry out Shapiro Test and Levene's Test on continuous variables.
        If a variable passes both tests, One_Way_F_test will be performed,
        otherwise, Kruskal test will be performed. 
        Input:
            group0: Dataframe for group 0
            group1: Dataframe for group 1
            alpha: Significance level
        Output:
            f_test: Dataframe(Feature: F test p value)
            kru_test: Dataframe(Feature: Kruskal test p value)
        """
        f_test = {}
        f_stats = {}
        kru_test = {}
        f_test_p = {}
        kru_stats = {}
        for i in self.cont_var:
            try:
                g0 = group0[i].dropna()
                g1 = group1[i].dropna()
                #Perform Shapirio Test on each group
                sha_p0 = stats.shapiro(g0)[1]
                sha_p1 = stats.shapiro(g1)[1]
                #Perfor Levene Test
                try:
                    L_p = stats.levene(g0,g1)[1]
                except ValueError as e:
                    print(e,'\n','Error occurs at column: ', i,'\n',g0,'\n',g1)
                #H0 is valid if p_value greater than alpha
                if (sha_p0 > alpha and sha_p1 > alpha and L_p > alpha):
                    f_test_p[i] = round(stats.f_oneway(g0,g1)[1],3)
                    f_stats[i] =  round(stats.f_oneway(g0,g1)[0],3)
                else:
                    kru_test[i] = round(stats.kruskal(g0,g1)[1],3)
                    kru_stats[i] = round(stats.kruskal(g0,g1)[0],3)
            except TypeError:
                print('TypeError: Feature ',i)
                break
        f_test = pd.DataFrame({'Statistics':f_stats,'P_value':f_test})
        kru_test = pd.DataFrame({'Statistics':kru_stats,'P_value':kru_test})
        return(f_test, kru_test)
                    
    def label_dist(self, df):
        distribution = df[self.label].value_counts()
        return(distribution)
    
    def encap_mean_std(self, x, decimal = 2):
        """
        Input: An array or a series
        Output: format: Mean(SD)
        """
        out = str(round(x['mean'],decimal)) + '(' + str(round(x['std'],decimal)) + ')'
        return(out)

    def encap_percentiles(self, x):
        """
        Input: An array or a series
        Output: format: median[25%,75%]
        """
        out = str(x['50%'])+'('+str(x['25%']) + ',' + str(x['75%']) + ')'
        return(out) 
    
    def cont_stats(self, df, group0, group1, stats_type = 'mean_std', decimal = 2):
        """
        Create statistical table including mean, std, and p_value for each case
        Input:
            df: The whole dataframe
            group0: Group 0 of df
            group1: Group 1 of df
            stats_type: String type, {'mean_std','percentiles'}
        Output:
            cont_stat: A dataframe including mean, std, and p_value for each case
        """
        desc_all = np.round(df[self.cont_var].describe(),decimal)
        desc_group0 = np.round(group0[self.cont_var].describe(),decimal)
        desc_group1 = np.round(group1[self.cont_var].describe(),decimal)
        if stats_type == 'mean_std':
            overall_stat = pd.DataFrame(desc_all.apply(lambda x: self.encap_mean_std(x, decimal)),columns = ['All'])
            g0_stat = pd.DataFrame(desc_group0.apply(lambda x: self.encap_mean_std(x, decimal)),columns = ['Group 0'])
            g1_stat = pd.DataFrame(desc_group1.apply(lambda x: self.encap_mean_std(x, decimal)),columns = ['Group 1'])
            cont_stat = pd.concat([overall_stat,g0_stat,g1_stat],axis = 1)
        elif stats_type == 'percentiles':
            overall_stat = pd.DataFrame(desc_all.apply(lambda x: self.encap_percentiles(x)),columns = ['All'])
            g0_stat = pd.DataFrame(desc_group0.apply(lambda x: self.encap_percentiles(x)),columns = ['Group 0'])
            g1_stat = pd.DataFrame(desc_group1.apply(lambda x: self.encap_percentiles(x)),columns = ['Group 1'])
            cont_stat = pd.concat([overall_stat,g0_stat,g1_stat],axis = 1)
        f,k = self.Norm_var_test(group0,group1)
        cont_stat = pd.concat([cont_stat,pd.concat([f,k])],axis = 1)
        cont_stat.columns = ['All' + '(N=' + str(df.shape[0]) + ')',
                             'Group 0' + '(n=' + str(group0.shape[0]) + ')',
                             'Group 1' + '(n=' + str(group1.shape[0]) + ')',
                             'Statistics',
                             'P_value']
        if f.shape[0] == 0:
            print('No variable is valid for parametric test')
        else:
            print('{} variables are valid for parametric test:\n'.format(f.shape[0]),f.index.tolist())
        return(cont_stat)
    
    def chi2_test(self,x,y):
        """
        Input:
            x: Categorical variables
            y: Label
        Output:
            p: Chi square p value
        """
        p = np.round(stats.chi2_contingency(pd.crosstab(x,y))[1],3)
        statistics = np.round(stats.chi2_contingency(pd.crosstab(x,y))[0],3)
        return(statistics,p)
    
    def encap_percent(self,x,sample_size,decimal = 2):
        """
        Input: An array or a series
        Output: format: count(percentage)
        """
        # print(x,sample_size)
        out = str(x) + '(' + str(np.round((x*100)/sample_size,decimal)) + '%' + ')'
        return(out)

    def cate_stat(self, df, group0, group1, decimal=2):
        """
        Input:
            df:Whole dataset
            group0: Group 0 of df
            group1: Group 1 of df
            decimal: Decimal precision
        Output:
            Statistical analysis of all categorical variables,
            corresponding p values are also presented.
            Binary variables and multi-category variables are included
        """
        cate_stat = pd.DataFrame()
        distribution = self.label_dist(df)
        # Binary statistics
        binary_cate = self.binary_var
        if len(binary_cate) != 0:
            binary_all_descr = pd.DataFrame(np.sum(df[binary_cate]),columns = ['All'])
            binary_g0_descr = pd.DataFrame(np.sum(group0[binary_cate]),columns = ['Group 0'])
            binary_g1_descr = pd.DataFrame(np.sum(group1[binary_cate]),columns = ['Group 1'])
            binary_cate_test = pd.DataFrame()
            binary_cate_test['Statistics'] = df[binary_cate].apply(lambda x:self.chi2_test(x,df[self.label])[0])
            binary_cate_test['P_value'] = df[binary_cate].apply(lambda x:self.chi2_test(x,df[self.label])[1])
            binary_stats = pd.concat([binary_all_descr, binary_g0_descr,binary_g1_descr,binary_cate_test],axis = 1)
        else:
            """
            2021/01/04 update:
                解决没有二分类时将index作为二计算的问题
            """
            binary_stats = pd.DataFrame()
        # Multi-category statistics
        if len(self.multicate_var) != 0:
            multi_cate_test = pd.DataFrame()
            multi_cate_test['Statistics'] = df[self.multicate_var].apply(lambda x:self.chi2_test(x,df[self.label])[0])
            multi_cate_test['P_value'] = df[self.multicate_var].apply(lambda x:self.chi2_test(x,df[self.label])[1])
                                                                                     
            cate_descr_full = pd.DataFrame()
            for i in self.multicate_var:
                cate_descr_header = pd.DataFrame([['']*3],columns = ['All','Group 0','Group 1'], index = [i])   
                cate_descr_body  = pd.concat([df.groupby(i)[i].size(),group0.groupby(i).size(),group1.groupby(i).size()],axis = 1)
                cate_descr_body.columns = ['All','Group 0','Group 1']
                cate_descr_body = cate_descr_body.fillna(0)
                cate_descr_full = cate_descr_full.append(pd.concat([cate_descr_header,cate_descr_body],axis = 0))
            cate_descr_full['P_value'] = np.NaN
            cate_descr_full.loc[multi_cate_test.index,'Statistics'] = multi_cate_test.loc[multi_cate_test.index,'Statistics']
            cate_descr_full.loc[multi_cate_test.index,'P_value'] = multi_cate_test.loc[multi_cate_test.index,'P_value']
            cate_descr_full.fillna('',inplace = True)
            full_cate_stats = binary_stats.append(cate_descr_full)
        else:
            full_cate_stats = binary_stats
        
        full_cate_stats.loc[full_cate_stats[full_cate_stats.All != ''].index,'All'] = full_cate_stats.\
            loc[full_cate_stats[full_cate_stats.All != ''].index,'All'].apply(lambda x: self.encap_percent(x,df.shape[0],decimal))
        full_cate_stats.loc[full_cate_stats[full_cate_stats.All != ''].index,'Group 0'] = full_cate_stats.\
            loc[full_cate_stats[full_cate_stats.All != ''].index,'Group 0'].apply(lambda x: self.encap_percent(x,distribution[0], decimal))
        full_cate_stats.loc[full_cate_stats[full_cate_stats.All != ''].index,'Group 1'] = full_cate_stats.\
            loc[full_cate_stats[full_cate_stats.All != ''].index,'Group 1'].apply(lambda x: self.encap_percent(x,distribution[1], decimal))
        full_cate_stats.columns = full_cate_stats.columns = ['All' + '(N=' + str(df.shape[0]) + ')',
                                                             'Group 0' + '(n=' + str(group0.shape[0]) + ')',
                                                             'Group 1' + '(n=' + str(group1.shape[0]) + ')',
                                                             'Statistics',
                                                             'P_value']
        return(full_cate_stats)

    def stats_table(self, df, group0, group1, stats_type='mean_std', decimal = 2):
        if len(self.cont_var) == 0:
            stats_table = self.cate_stat(df, group0, group1, decimal = decimal)
        elif len(self.cate_var) == 0:
            stats_table = self.cont_stats(df,group0, group1, stats_type = stats_type)
        else:
            stats_table = pd.concat([self.cont_stats(df,group0, group1, stats_type = stats_type),
                                    self.cate_stat(df, group0, group1, decimal = decimal)])
        stats_table['P_value'] = stats_table['P_value'].replace({0: '<0.001'})
        return stats_table


def p_sele(full_stat):
    '''
    p-value<0.05
    '''
    def check_significance(x):
        try:
            if x == '<0.001':
                return 1
            if x < 0.05:
                return 1
            else:
                return 0
        except:
            return np.nan

    full_stat['是否显著'] = full_stat['P_value'].apply(lambda x: check_significance(x))
    return full_stat[full_stat['是否显著'] == 1].index.tolist()


def base_sta(train, test, classes=5):
    """
    Created on Fri Apr  2 09:26:48 2021
    数据集之间的假设检验
    注意：有缺失值的时候，跑出来的P-value是NAN
    @author: FengY Z
    """
    compared_vars = [n for n in train.columns if len(train[n].value_counts()) <= classes]
    cont_var = [n for n in train.columns if len(train[n].value_counts()) > classes]

    p_value_dict = {}
    all_df = pd.DataFrame()
    for var in compared_vars:
        train_fenbu = train[var].value_counts().reset_index()
        train_fenbu.rename(columns={var: 'train'}, inplace=True)
        train_fenbu_zhanbi = train[var].value_counts(normalize=True).reset_index()
        train_fenbu_zhanbi.rename(columns={var: 'train_rate'}, inplace=True)
        test_fenbu = test[var].value_counts().reset_index()
        test_fenbu.rename(columns={var: 'test'}, inplace=True)
        test_fenbu_zhanbi = test[var].value_counts(normalize=True).reset_index()
        test_fenbu_zhanbi.rename(columns={var: 'test_rate'}, inplace=True)
        for df in [train_fenbu_zhanbi, test_fenbu, test_fenbu_zhanbi]:
            train_fenbu = pd.merge(train_fenbu, df, on='index', how='outer')
        train_fenbu.rename(columns={'index': 'Subtype'}, inplace=True)
        train_fenbu['Characteristics'] = [var] * len(train_fenbu)
        all_df = all_df.append(train_fenbu)
        ks_test, p_value = ks_2samp(train[var], test[var])
        p_value_dict[var] = [p_value]

    all_df['train_rate'] = all_df['train_rate'].apply(lambda x: '(' + str(round(x * 100, 2)) + '%' + ')')
    all_df['test_rate'] = all_df['test_rate'].apply(lambda x: '(' + str(round(x * 100, 2)) + '%' + ')')
    all_df['train'] = all_df['train'].apply(lambda x: str(x))
    all_df['test'] = all_df['test'].apply(lambda x: str(x))
    all_df['train'] = all_df['train'] + all_df['train_rate']
    all_df['test'] = all_df['test'] + all_df['test_rate']
    all_df.drop(['train_rate', 'test_rate'], axis=1, inplace=True)
    p_value_df = pd.DataFrame(p_value_dict).T.reset_index()
    p_value_df.rename(columns={'index': 'Characteristics', 0: 'p_values'}, inplace=True)
    all_df = all_df.merge(p_value_df, on='Characteristics', how='left')
    all_df = all_df[['Characteristics', 'Subtype', 'train', 'test', 'p_values']]
    new_all_df = pd.DataFrame()
    for df in all_df.groupby('Characteristics'):
        tmp = df[1].sort_values(by='Subtype')
        new_all_df = new_all_df.append(tmp)
    result_dict = {}

    for var in cont_var:
        train_mean = train[var].mean()
        test_mean = test[var].mean()

        train_std = train[var].std()
        test_std = test[var].std()

        _, levene_p = levene(train[var], test[var])
        #         print(levene_p)
        if levene_p < 0.05:
            _, p_value = ttest_ind(train[var], test[var], equal_var=False)
        else:
            _, p_value = ttest_ind(train[var], test[var])
        result_dict[var] = ["{}({})".format(round(train_mean, 2), round(train_std, 2)),
                            "{}({})".format(round(test_mean, 2), round(test_std, 2)), round(p_value, 3)]

    con_df = pd.DataFrame(result_dict).T.reset_index()
    con_df.rename(columns={'index': 'Characteristics', 0: 'train', 1: 'test', 2: 'p_values'}, inplace=True)
    con_df['Subtype'] = [np.nan] * len(con_df)
    con_df = con_df[['Characteristics', 'Subtype', 'train', 'test', 'p_values']]
    sta_df = pd.concat([new_all_df, con_df], axis=0)
    return sta_df

