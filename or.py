from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model,save_plot
import pandas as pd

OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}
df = pd.DataFrame(OR)
print(df)

X,y = prepare_data(df)
ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()


save_model(model=model,filename='or.model')
save_plot(df=df,file_name='or.png',model=model)