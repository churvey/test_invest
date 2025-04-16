from catboost import CatBoostRegressor, Pool
from trade.model.model import Model

class RegCat(Model):

    def __init__(
        self, features, **kwargs
    ):  
        super(RegCat, self).__init__(features)
        self.model = CatBoostRegressor(
                            **kwargs,
                        )
        
    def forward(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
