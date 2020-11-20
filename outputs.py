import pandas as pd


def make_output(model, test_data, ids, file_name='submission.csv'):
    pred = model.predict(test_data)
    
    out = pd.DataFrame(pred, columns=['Class_' + str(i) for i in range(1, 10)])
    out.insert(loc=0, column='id', value=ids)
    out.to_csv(file_name, index=False)
    print(f'Written submission file to: {file_name}')
