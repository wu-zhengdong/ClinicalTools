import os
import datetime
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt
import shap
shap.initjs()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['axes.unicode_minus'] = False

class SHAP:
    '''
    # 1. 树模型用 TreeExplainer
    # 2. 其他模型用 KernelExplainer
    # 3. 对最佳的模型进行SHAP就行
    '''
    def __init__(self, model, x_raw, x, y_test, y_pred, explainer='kernel'):
        self.x_test_raw = x_raw
        self.x_test = x
        self.model = model

        # y_test: dataframe
        self.classified = y_test[y_test.values.reshape(-1,) == y_pred].index  # 匹配正确的病人索引
        self.misclassified = y_test[y_test.values.reshape(-1,) != y_pred].index  # 匹配错误的病人索引

        if explainer == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict_proba, self.x_test, link='logit')
        if explainer == 'tree':
            self.explainer = shap.TreeExplainer(self.model, )
        print('SHAP is starting...')
        start = datetime.datetime.now().replace(microsecond=0)
        self.shap_values = self.explainer.shap_values(x, nsamples=100, random_state=0)
        end = datetime.datetime.now().replace(microsecond=0)
        print('Finish! Consume Time: {}\n'.format(end-start))

    def summary_plot(self, save_path='./results/shap/summary/'):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for t in ['bar', 'dot', 'violin', 'layered_violin']:
            fig, ax = plt.subplots(figsize=(10, 10))
            fig = shap.summary_plot(self.shap_values[1], self.x_test_raw, show=False, plot_type=t)
            ax.set_title("Shap Value Feature Importance", size=25, fontdict={'weight': 'bold'})
            ax.yaxis.set_tick_params(labelsize=18)
            ax.xaxis.set_tick_params(labelsize=18)
            ax.set_xlabel('SHAP value(impact on model output)', size=18)
            plt.savefig(os.path.join(save_path, 'summary_{}.png'.format(t)), dpi=300, bbox_inches='tight')
            plt.tight_layout()

    def dependence_plot(self, save_path='./results/shap/dependence/'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 可以直接循环跑，每个变量的结果
        for n in self.x_test_raw.columns:
            shap.dependence_plot(n, self.shap_values[1], self.x_test_raw, interaction_index=None, show=False)
            plt.savefig(os.path.join(save_path, '{}.png'.format(n.split('(')[0])), dpi=300, bbox_inches='tight')

    def force_plot(self, save_path='./results/shap/force/html'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        classified_force_plot_path = os.path.join(save_path, 'classified')
        misclassified_force_plot_path = os.path.join(save_path, 'misclassified')

        # 保存 png 的路径，在 selenium 模块用，这里一并创建
        classified_save_png_path = os.path.join(save_path[:-4]+'png', 'classified')
        misclassified_save_png_path = os.path.join(save_path[:-4]+'png', 'misclassified')

        if not os.path.exists(classified_force_plot_path):
            os.makedirs(classified_force_plot_path)
            os.makedirs(misclassified_force_plot_path)

            # 保存 png 的路径，在 selenium 模块用，这里一并创建
            os.makedirs(classified_save_png_path)
            os.makedirs(misclassified_save_png_path)

        for i in tqdm(range(len(self.classified))):
            f = shap.force_plot(self.explainer.expected_value[1], self.shap_values[1][self.classified[i]],
                                features=self.x_test_raw.iloc[self.classified[i]],
                                feature_names=self.x_test_raw.columns,
                                out_names='model output value',
                                link='logit')
            shap.save_html(
                os.path.join(classified_force_plot_path, 'force_plot_patients_{}.html'.format(self.classified[i])), f)
        for i in tqdm(range(len(self.misclassified))):
            f = shap.force_plot(self.explainer.expected_value[1], self.shap_values[1][self.misclassified[i]],
                                features=self.x_test_raw.iloc[self.misclassified[i]],
                                feature_names=self.x_test_raw.columns,
                                out_names='model output value',
                                link='logit')
            shap.save_html(
                os.path.join(misclassified_force_plot_path, 'force_plot_patients_{}.html'.
                             format(self.misclassified[i])), f)

    def decision_plot(self, save_path='./results/shap/decision'):
        '''
        feature_order 的排序是倒叙，建议跑出默认顺序后，用一张记录一下每个变量对应的序号。
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        classified_decision_plot_path = os.path.join(save_path, 'classified')
        misclassified_decision_plot_path = os.path.join(save_path, 'misclassified')

        if not os.path.exists(classified_decision_plot_path):
            os.makedirs(classified_decision_plot_path)
            os.makedirs(misclassified_decision_plot_path)

        for i in range(len(self.classified)):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title("Patient {} Decision Plot".format(self.classified[i]), size=25, fontdict={'weight': 'bold'})
            #     ax.set_yticklabels(size = 23,fontdict = {'family':'Times New Roman'})
            ax.yaxis.set_tick_params(labelsize=20)
            ax.xaxis.set_tick_params(labelsize=20)
            ax.set_xlabel('Model output value', size=20)
            plt.tight_layout()
            fig = shap.decision_plot(self.explainer.expected_value[1],
                                     self.shap_values[1][self.classified[i]],
                                     self.x_test_raw.iloc[self.classified[i]],
                                     # feature_order=[0, 11, 12, 3, 5, 2, 4, 13, 7, 6, 10, 8, 9, 1],  # 排序
                                     #                              return_objects=True,
                                     show=False,
                                     link='logit',
                                     highlight=0)
            plt.savefig(os.path.join(classified_decision_plot_path, 'patient_{}.png'.format(self.classified[i])), dpi=300,
                        bbox_inches='tight')

        for i in range(len(self.misclassified)):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title("Patient {} Decision Plot".format(self.misclassified[i]), size=25, fontdict={'weight': 'bold'})
            ax.yaxis.set_tick_params(labelsize=20)
            ax.xaxis.set_tick_params(labelsize=20)
            ax.set_xlabel('Model output value', size=20)
            plt.tight_layout()
            fig = shap.decision_plot(self.explainer.expected_value[1],
                                     self.shap_values[1][self.misclassified[i]],
                                     self.x_test_raw.iloc[self.misclassified[i]],
                                     # feature_order=[0, 11, 12, 3, 5, 2, 4, 13, 7, 6, 10, 8, 9, 1],
                                     #                              return_objects=True,
                                     show=False,
                                     link='logit',
                                     highlight=0)
            plt.savefig(os.path.join(misclassified_decision_plot_path, 'patient_{}.png'.format(self.misclassified[i])),
                        dpi=300, bbox_inches='tight')
