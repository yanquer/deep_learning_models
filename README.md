# deep_learning_models

## 前言

个人看来，由于科技的局限性，机器真正认识的只有二进制0与1，
所以深度学习的本质在于寻找一个适用性强的函数。

另一方面，由于现在科技发展迅速，机器有能力处理大批量数据，
某些函数的结合无法有效单无理论支持，很难证明为什么这个函数/方法是有效的，
从这方面看，深度学习也具有了一些哲学与玄学的色彩。


## 环境说明

#### 介绍
此仓库主要记录学习过程中自己训练的模型，

#### 软件架构
python

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)




## 结构说明

- models
  - defs 一些公共宏定义
  - DNN 多层神经网络模版定义
  - utils 工具函数定义
  - cat 重构的识别猫的代码 新的代价函数 ![cost](./images/img_cat_cost.png)
  - flower 重构的时候发现一个bug
    感觉是python底层解析执行的问题，向量化计算会失败, 出bug代码 ![bug_code](./images/bug/bug_code.png)
    bug示例，部分截图 ![bug_res](./images/bug/bug_res.png)
    有意思的是，专门写了一个测试把原有数据拿去测试，居然复现不出 ![bug_test](./images/bug/bug_test.png)
    排查到凌晨，发现是这个问题。。。具体原因还没找到，只找到bug点。
    
    输出放到 log文件夹下面


- old_models 一些旧的模型（弃用了可以算）
  - pubfun 定义一些公共的函数
  - single_neuron 单神经元逻辑回归模型，识别图片是否是猫，训练损失率大概为 ![训练识别猫模型](./images/img.png)
  - mul_neural 一个隐藏层4个神经元的浅层DNN模型，计算损失率为![](./images/img_flower.png)


