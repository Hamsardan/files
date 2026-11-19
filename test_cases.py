"""Exactly 20 independently graded checks (3 marks each = 60 marks)."""
import os
from pathlib import Path
import joblib, nbformat, numpy as np, pandas as pd, pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent

@pytest.fixture(scope="module")
def app():
    notebook_name = os.environ.get("MS2_NOTEBOOK", "solution.ipynb")
    notebook_path = BASE / notebook_name
    notebook = nbformat.read(notebook_path, as_version=4)
    module = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and "definitions" in cell.metadata.get("tags", []):
            if cell.source.strip():
                exec(compile(cell.source, str(notebook_path), "exec"), module)
    return module

def shipment_rows(n=4):
    return pd.DataFrame({
        "shipment_id": [f"SH-{i:04d}" for i in range(n)],
        "dispatch_ts": ["2026-01-03 04:15", "04/02/2026 13:20", "2026-03-05 22:45", "06/04/2026 08:10"][:n],
        "distance_km": [18.75, 21.0, 30.5, 42.0][:n],
        "weight_kg": [3.2, 4.5, 1.25, 2.0][:n],
        "month_text": ["Jan", "FEB", " mar ", "DEC"][:n],
        "transit_hours": [2.0, 3.0, 0.0, 4.0][:n],
        "temperature_c": [5.0, 8.0, 2.0, 10.0][:n],
        "region": [" North ", "SOUTH", "West", "East"][:n],
        "vehicle_type": [" VAN ", "truck", "Bike", "VAN"][:n],
        "route_zone": ["2.0", "1.0", "3.0", "4.0"][:n],
        "late_delivery": ["yes", "no", " YES ", "NO"][:n],
    })

def write_frame(path, df):
    df.to_csv(path, index=False); return str(path)

def feature_frame(app, n=12):
    data={}
    for i,name in enumerate(app["NUMERIC"]): data[name]=np.arange(n,dtype=float)+i+1
    for name in app["CATEGORICAL"]: data[name]=(["north","south","west"]*(n//3))+["north"]*(n%3)
    return pd.DataFrame(data,columns=app["FEATURES"])

def test_T01_first_duplicate_id_is_retained(app,tmp_path):
    df=shipment_rows(3); df.loc[1,"shipment_id"]=df.loc[0,"shipment_id"]; df.loc[1,"temperature_c"]=99
    result=app["load_data"](write_frame(tmp_path/"duplicates.csv",df))
    assert result["shipment_id"].tolist()==["SH-0000","SH-0002"] and result.loc[0,"temperature_c"]==5

def test_T02_remove_invalid_mixed_timestamps(app,tmp_path):
    df=shipment_rows(3); df["dispatch_ts"]=["2026-03-07 22:30","08/03/2026 09:15","not-a-date"]
    result=app["load_data"](write_frame(tmp_path/"dates.csv",df))
    assert len(result)==2 and result["dispatch_ts"].dt.hour.tolist()==[22,9]

def test_T03_numeric_coercion_and_infinity_replacement(app,tmp_path):
    df=shipment_rows(4); df["transit_hours"]=["bad","inf","-inf","2.5"]; df["temperature_c"]=["1.5","bad","inf","-inf"]
    result=app["load_data"](write_frame(tmp_path/"numeric.csv",df))
    assert all(pd.api.types.is_numeric_dtype(result[c]) for c in app["RAW_NUMERIC"])
    assert not np.isinf(result[app["RAW_NUMERIC"]].to_numpy(dtype=float)).any()

def test_T04_normalize_all_categorical_features(app,tmp_path):
    df=shipment_rows(3); df.loc[0,["region","vehicle_type","route_zone"]]=["  NORTH "," VAN ","2.0"]; df.loc[1,["region","vehicle_type","route_zone"]]=[" ",None,"  "]
    result=app["load_data"](write_frame(tmp_path/"cat.csv",df))
    assert result.loc[0,"region"]=="north" and result.loc[1,"region"]=="unknown" and result.loc[1,"vehicle_type"]=="unknown"

def test_T05_derive_dispatch_hour(app,tmp_path):
    result=app["load_data"](write_frame(tmp_path/"hours.csv",shipment_rows(4)))
    assert result["departure_hour"].tolist()==[4,13,22,8]

def test_T06_distance_numeric_coercion(app,tmp_path):
    df=shipment_rows(4); df["distance_km"]=["18.75","bad","inf","-inf"]
    result=app["load_data"](write_frame(tmp_path/"distance.csv",df))
    assert result.loc[0,"distance_km"]==pytest.approx(18.75) and result["distance_km"].isna().tolist()==[False,True,True,True]

def test_T07_weight_numeric_coercion(app,tmp_path):
    df=shipment_rows(4); df["weight_kg"]=["3.25","bad","inf","-inf"]
    result=app["load_data"](write_frame(tmp_path/"weight.csv",df))
    assert result.loc[0,"weight_kg"]==pytest.approx(3.25) and result["weight_kg"].isna().tolist()==[False,True,True,True]

def test_T08_clean_numeric_like_route_codes(app,tmp_path):
    result=app["load_data"](write_frame(tmp_path/"route.csv",shipment_rows(4)))
    assert result["route_zone"].tolist()==["2","1","3","4"]

def test_T09_map_month_strings_jan_to_dec(app,tmp_path):
    df=pd.concat([shipment_rows(4)]*3,ignore_index=True); df["shipment_id"]=[f"SH-{i:04d}" for i in range(12)]; df["month_text"]=["Jan","FEB","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","DEC"]
    result=app["load_data"](write_frame(tmp_path/"months.csv",df))
    assert result["month_number"].tolist()==list(range(1,13))

def test_T10_derive_hour_from_parsed_timestamp(app,tmp_path):
    result=app["load_data"](write_frame(tmp_path/"parsed.csv",shipment_rows(4)))
    assert result["departure_hour"].tolist()==result["dispatch_ts"].dt.hour.tolist()

def test_T11_safe_distance_per_hour(app,tmp_path):
    result=app["load_data"](write_frame(tmp_path/"ratio.csv",shipment_rows(4)))
    assert result.loc[0,"distance_per_hour"]==pytest.approx(9.375) and pd.isna(result.loc[2,"distance_per_hour"])

def test_T12_exact_feature_groups(app):
    assert len(app["NUMERIC"])==7 and len(app["CATEGORICAL"])==3 and app["FEATURES"]==app["NUMERIC"]+app["CATEGORICAL"]

def test_T13_column_transformer_two_branches(app):
    p=app["build_pipeline"](LogisticRegression()); t=p.named_steps["preprocessor"]
    assert isinstance(p,Pipeline) and isinstance(t,ColumnTransformer) and [x[0] for x in t.transformers]==["numeric","category"]

def test_T14_unseen_categories_after_training(app):
    X=feature_frame(app); y=pd.Series([0,1]*6); m=app["build_pipeline"](LogisticRegression()); m.fit(X,y); new=X.iloc[:2].copy(); new.loc[new.index[0],app["CATEGORICAL"][0]]="never-seen"
    assert m.predict(new).shape==(2,)

def test_T15_map_yes_no_and_one_zero(app,tmp_path):
    tr=shipment_rows(4); tr["late_delivery"]=[" YES ","no","1","0"]; p=write_frame(tmp_path/"train.csv",tr)
    _,_,yt,ye=app["prepare_datasets"](p,p); assert yt.tolist()==[1,0,1,0] and ye.tolist()==[1,0,1,0]

def test_T16_fit_separate_balanced_models(app):
    X=feature_frame(app,24); y=pd.Series([0,1]*12); models=app["train_models"](X,y)
    assert set(models)=={"logistic_regression","random_forest"}; lr=models["logistic_regression"].named_steps["model"]; rf=models["random_forest"].named_steps["model"]
    assert isinstance(lr,LogisticRegression) and isinstance(rf,RandomForestClassifier) and lr.class_weight==rf.class_weight=="balanced"

def test_T17_positive_class_f1_threshold(app):
    class Fake:
        classes_=np.array([0,1])
        def __init__(self,p): self.p=np.asarray(p)
        def predict_proba(self,X): return np.column_stack([1-self.p,self.p])
    y=pd.Series([1,1,0,0]); X=pd.DataFrame({"v":range(4)}); models={"alpha":Fake([.8,.8,.1,.1]),"gamma":Fake([.8,.8,.1,.1]),"beta":Fake([.8,.1,.1,.1]),"boundary":Fake([.40,.39,.40,.90])}
    c=app["evaluate_models"](models,X,y)
    assert c.columns.tolist()==["model_name","accuracy","precision","recall","f1"] and c["model_name"].tolist()==["alpha","gamma","beta","boundary"]
    assert np.isclose(c.set_index("model_name").loc["boundary","f1"],f1_score(y,[1,0,1,1]))

def test_T18_select_existing_best(app):
    c=pd.DataFrame({"model_name":["alpha","gamma","beta"],"f1":[.8,.8,.5],"recall":[.9,.9,.7]}); models={k:object() for k in ["alpha","beta","gamma"]}
    name,m=app["select_best_model"](c,models); assert name=="alpha" and m is models["alpha"]

def test_T19_prediction_schema_threshold(app):
    class Fake:
        classes_=np.array([0,1])
        def predict_proba(self,X):
            p=np.array([.10,.39,.40,.95]); return np.column_stack([1-p,p])
    df=feature_frame(app,4); df["shipment_id"]=["S1","S2","S3","S4"]; r=app["predict_delay"](Fake(),df)
    assert r.columns.tolist()==["shipment_id","predicted_late","late_probability"] and r["predicted_late"].tolist()==[0,0,1,1] and r["late_probability"].tolist()==[.1,.39,.4,.95]

def test_T20_train_predict_write_three_files(app,tmp_path,monkeypatch):
    for name,n in [("shipment_train.csv",80),("shipment_test.csv",24),("shipment_predict.csv",12)]: pd.read_csv(BASE/name).iloc[:n].to_csv(tmp_path/name,index=False)
    monkeypatch.chdir(tmp_path); name,c=app["train"](); assert name in c["model_name"].tolist(); assert (tmp_path/"best_model.pkl").is_file() and (tmp_path/"model_comparison.csv").is_file() and isinstance(joblib.load(tmp_path/"best_model.pkl"),Pipeline)
    r=app["predict"](); assert len(r)==12 and (tmp_path/"shipment_predictions.csv").is_file()
