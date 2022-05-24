from sklearn.base import BaseEstimator, RegressorMixin


class traffic_predictor(BaseEstimator, RegressorMixin):
    def __init__(self, precipitacion, viento, estacion, dia_semana, mes, dia, hora, prediction, interactions):
        self.dia = dia
        self.dia_semana = dia_semana
        self.precipitacion = precipitacion
        self.viento = viento
        self.estacion = estacion
        self.mes = mes
        self.hora = hora
        self.prediction = prediction
        self.interactions = interactions
        self.pipeline = predicting_pipeline(precipitacion, viento, estacion, dia_semana, mes, dia, hora, prediction, interactions)
        self._estimator_type = "regressor"
    def fit(self, X, y):
        self.pipeline.fit(X, y)
        #return self
    def predict(self, X):
        return self.pipeline.predict(X)
    def transform(self, X):
        return self.pipeline.transform(X)
    def set_params(self, **parameters):
        for a in ["precipitacion", "viento", "estacion", "dia_semana", "mes", "dia", "hora", "prediction", "interactions"]:
            if a in parameters.keys():
                setattr(self, a, parameters[a])
        #BaseEstimator.set_params(self, **parameters)
        self.pipeline = predicting_pipeline(self.precipitacion, self.viento, self.estacion, self.dia_semana, self.mes, self.dia, self.hora, self.prediction, self.interactions)
        #for parameter, value in parameters.items():
        #    setattr(self, parameter, value)
        return self
    def score(self, X, y):
        return self.pipeline.score(X, y)