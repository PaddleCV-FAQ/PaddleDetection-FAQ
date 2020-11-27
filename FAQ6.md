#### 6.如何将训练的记录保存成txt
①在 `./tools/train.py` 中写入 write() 函数
```python
def write(logstats):
    filename = 'log.txt'
    with open(filename,'a',encoding='utf-8') as f:
        f.writelines(logstats + '\n')
        f.close
```
② 找到此段代码，加入 `write(strs)` 代码
```python
if it % cfg.log_iter == 0 and (not FLAGS.dist or trainer_id == 0):
    strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}, eta: {}'.format(
        it, np.mean(outs[-1]), logs, time_cost, eta)
    logger.info(strs)
    write(strs)
```
![](https://ai-studio-static-online.cdn.bcebos.com/b99c6becb51b47ccb85068865f685b4ef1504faa11054e118cdfedb6806cf937)

