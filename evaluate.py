from sklearn.metrics import f1_score, accuracy_score

def Evaluatation(args, outputs, targets):
    eval_dicts = {}
    
    if 'f1-score' in args.metric_score:
        f1 = f1_score(targets, outputs, average='macro')
        eval_dicts['f1-score'] = f1
    
    if 'acc-score' in args.metric_score:
        acc = accuracy_score(targets, outputs)
        eval_dicts['acc-score'] = acc
        
    return eval_dicts