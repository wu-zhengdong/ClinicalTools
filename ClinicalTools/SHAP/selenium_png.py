from selenium import webdriver
import time
import re
import os


def fetchId(file):
    html = open(file, encoding='utf-8').read()
    divId = re.findall("<div id='(.*?)'>", html)[0]
    return divId


def screenshot(file, screenshot_png, divId,
               chromedriver):
    browser = webdriver.Chrome(chromedriver)  # 启动 selenium
    browser.get(file)
    browser.maximize_window()
    time.sleep(1)
    idElement = browser.find_element_by_id(divId)
    idElement.screenshot(screenshot_png)
    browser.quit()


def run(root_path='./results/shap/force/html/',
        save_png_path='./results/shap/force/png/',
        chromedriver='./chromedriver.exe'):
    '''
    :param root_path:  SHAP 绘制的 HTML 文件路径
    :param save_png_path: HTML to PNG 后的文件路径
    :return: None
    '''
    if not os.path.exists(save_png_path):
        os.makedirs(save_png_path)

    # 启动 Classified 文件（分类正确的病人）
    for f in os.listdir(root_path):
        # 保存路径
        pngFile = os.path.join(save_png_path, f.split('.')[0] + '.png')
        # 使用正则提取 div 元素的 ID
        re_file = os.path.join(root_path, f)
        divId = fetchId(re_file)
        # 使用 selenium 截取 div 元素的图片，需要注意的是，这里必须要用全路径
        selenium_file = os.path.join(os.getcwd(), re_file)
        screenshot(selenium_file, pngFile, divId, chromedriver)


def html2png(classified_force_plot_path='./results/shap/force/html/classified',
             misclassified_force_plot_path='./results/shap/force/html/misclassified',
             classified_save_png_path='./results/shap/force/png/classified',
             misclassified_save_png_path='./results/shap/force/png/misclassified',
             chromedriver='./chromedriver.exe'):
    '''
    This func can convert html to png.
    When you wanna to start this func, pls wait patiently.
    '''
    run(classified_force_plot_path, classified_save_png_path, chromedriver)
    run(misclassified_force_plot_path, misclassified_save_png_path, chromedriver)