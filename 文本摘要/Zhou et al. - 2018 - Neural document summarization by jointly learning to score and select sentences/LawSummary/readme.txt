该项目代码来源:https://github.com/magic282/NeuSum
data中为起诉意见书文本摘要语料，其中包含训练集(train)、验证集(dev)、测试集(test)。
src表示起诉书原文，句子间以##SENT##隔开；tgt表示参考摘要，来源于原文的句子。
*.rouge_bigram_F1、*.rouge_bigram_F1.regGain文件为通过起诉书原文和参考摘要计算得到的rouge值,通过运行cnndm_acl18中的程序得到。
cnndm_acl18来源：https://github.com/magic282/cnndm_acl18
训练：LawSummary\model_1\NeuSum\run_model.sh
预测：LawSummary\model_1\NeuSum\predict.sh
评价：LawSummary\model_1\NeuSum\evalution.sh