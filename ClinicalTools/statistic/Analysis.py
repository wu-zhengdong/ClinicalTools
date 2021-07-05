# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 09:13:53 2021

@author: MZH
"""
import pandas as pd
from scipy import stats
import numpy as np


class Aid:
    def __init__(self, label, dataframe, class_threshold = 10, multicate_var = None):
        """
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
    

    def stats_table(self, df, group0, group1, stats_type = 'mean_std', decimal = 2):
        if len(self.cont_var) == 0:
            stats_table = self.cate_stat(df, group0, group1, decimal = decimal)
        elif len(self.cate_var) == 0:
            stats_table = self.cont_stats(df,group0, group1, stats_type = stats_type)
        else:
            stats_table = pd.concat([self.cont_stats(df,group0, group1, stats_type = stats_type),
                                    self.cate_stat(df, group0, group1, decimal = decimal)])
        stats_table['P_value'] = stats_table['P_value'].replace({0:'<0.001'})
        return stats_table


    

    
    
    
    
    
    
    
    
    
    
    
    


















