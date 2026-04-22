import numpy as np
import pandas as pd

def generate_bn_insurance(n=1000, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    # Exógenas
    Age = np.random.normal(0, 1, n) # Linear
    SocioEcon = np.random.normal(0.5 * np.sin(Age) + 0.3 * Age**2, 1, n) # Non-Linear
    RiskAversion = np.random.normal(np.tanh(Age) + 0.5 * SocioEcon, 1, n) # Non-Linear
    MakeModel = np.random.normal(3 + 0.5 * RiskAversion + 0.6 * SocioEcon, 1, n) # Linear
    Mileage = np.random.normal(0, 1, n) # Linear
    HomeBase = np.random.normal(0.3 * RiskAversion + 0.5 * SocioEcon, 1, n) # Linear
    SeniorTrain = np.random.normal(0.3 * Age + np.sin(RiskAversion), 1, n) # Non-Linear
    VehicleYear = np.random.normal(0.6 * RiskAversion + 0.6 * SocioEcon, 1, n) # Linear
    Airbag = np.random.normal(np.tanh(MakeModel) + 0.5 * VehicleYear, 1, n) # Non-Linear
    GoodStudent = np.random.normal(0.6 * Age - 0.4 * SocioEcon, 1, n) # Linear
    DrivingSkill = np.random.normal(1 + np.tanh(SeniorTrain) + 0.4 * Age, 1, n) # Non-linear
    Cushioning = np.random.normal(0.8 * Airbag, 1, n) # Linear
    DrivQuality = np.random.normal(0.7 * RiskAversion + 0.6 * DrivingSkill,1, n) # Linear
    Antilock = np.random.normal(np.sin(VehicleYear + 0.5 * MakeModel), 1, n) # Non-Linear
    CarValue = np.random.normal(np.tanh(MakeModel) + 0.4 * Mileage + 0.2 * VehicleYear, 1, n) # Nonlinear
    AntiTheft = np.random.normal(1 / (1 + np.exp(-SocioEcon)) + RiskAversion**2, 1, n) # Nonlinear
    Accident = np.random.normal(np.sin(0.5 * Antilock + 0.4 * Mileage + 0.4 * DrivQuality), 1, n) # Nonlinear
    Theft = np.random.normal(0.6 * AntiTheft + 0.5 * HomeBase + 0.4 * CarValue, 1, n) # Linear
    RuggedAuto = np.random.normal(0.8 * VehicleYear + 0.5 * MakeModel, 1, n) # Linear
    ThisCarDam = np.random.normal(0.8 * Accident + 0.3 * RuggedAuto, 1, n) # Linear
    OtherCar = np.random.normal(0.7 * SocioEcon, 1, n) # Linear
    MedCost = np.random.normal(0.6 * Accident + 0.5 * Cushioning + np.sin(Age), 1, n) # Non-Linear
    ThisCarCost = np.random.normal(0.6 * ThisCarDam + 0.5 * Theft + 0.4 * CarValue, 1, n) # Linear
    OtherCarCost = np.random.normal(0.8 * Accident + 0.5 * RuggedAuto, 1, n) # Linear
    ILiCost = np.random.normal(0.7 * Accident, 1, n) # Linear
    DrivHist = np.random.normal(1 + np.tanh(DrivingSkill), 1, n) # Nonlinear
    PropCost = np.random.normal(0.8 * ThisCarCost + 0.5 * OtherCar, 1, n) # Linear

    df = pd.DataFrame({
        "Age": Age,
        "SocioEcon": SocioEcon,
        "GoodStudent": GoodStudent,
        "RiskAversion": RiskAversion,
        "DrivHist": DrivHist,
        "RuggedAuto": RuggedAuto,
        "PropCost": PropCost,
        "VehicleYear": VehicleYear,
        "MakeModel": MakeModel,
        "DrivQuality": DrivQuality,
        "Mileage": Mileage,
        "Antilock": Antilock,
        "DrivingSkill": DrivingSkill,
        "SeniorTrain": SeniorTrain,
        "ThisCarDam": ThisCarDam,
        "Theft": Theft,
        "CarValue": CarValue,
        "HomeBase": HomeBase,
        "AntiTheft": AntiTheft,
        "ThisCarCost": ThisCarCost,
        "OtherCarCost": OtherCarCost,
        "OtherCar": OtherCar,
        "MedCost": MedCost,
        "Cushioning": Cushioning,
        "Airbag": Airbag,
        "ILiCost": ILiCost,
        "Accident": Accident
    })

    return df


def ground_truth_insurance(df):
    # Extraer variables
    Age = df['Age'].values
    SocioEcon = df['SocioEcon'].values
    RiskAversion = df['RiskAversion'].values
    MakeModel = df['MakeModel'].values
    Mileage = df['Mileage'].values
    HomeBase = df['HomeBase'].values
    SeniorTrain = df['SeniorTrain'].values
    VehicleYear = df['VehicleYear'].values
    Airbag = df['Airbag'].values
    GoodStudent = df['GoodStudent'].values
    DrivingSkill = df['DrivingSkill'].values
    Cushioning = df['Cushioning'].values
    DrivQuality = df['DrivQuality'].values
    Antilock = df['Antilock'].values
    CarValue = df['CarValue'].values
    AntiTheft = df['AntiTheft'].values
    Accident = df['Accident'].values
    Theft = df['Theft'].values
    RuggedAuto = df['RuggedAuto'].values
    ThisCarDam = df['ThisCarDam'].values
    OtherCar = df['OtherCar'].values
    MedCost = df['MedCost'].values
    ThisCarCost = df['ThisCarCost'].values
    OtherCarCost = df['OtherCarCost'].values
    ILiCost = df['ILiCost'].values
    DrivHist = df['DrivHist'].values
    PropCost = df['PropCost'].values

    # Log-probabilidades
    logp_Age = -0.5*np.log(2*np.pi) - 0.5*(Age**2)
    logp_SocioEcon = -0.5*np.log(2*np.pi) - 0.5*((SocioEcon - (0.5*np.sin(Age) + 0.3*Age**2))**2)
    logp_RiskAversion = -0.5*np.log(2*np.pi) - 0.5*((RiskAversion - (np.tanh(Age) + 0.5*SocioEcon))**2)
    logp_MakeModel = -0.5*np.log(2*np.pi) - 0.5*((MakeModel - (3 + 0.5*RiskAversion + 0.6*SocioEcon))**2)
    logp_Mileage = -0.5*np.log(2*np.pi) - 0.5*(Mileage**2)
    logp_HomeBase = -0.5*np.log(2*np.pi) - 0.5*((HomeBase - (0.3*RiskAversion + 0.5*SocioEcon))**2)
    logp_SeniorTrain = -0.5*np.log(2*np.pi) - 0.5*((SeniorTrain - (0.3*Age + np.sin(RiskAversion)))**2)
    logp_VehicleYear = -0.5*np.log(2*np.pi) - 0.5*((VehicleYear - (0.6*RiskAversion + 0.6*SocioEcon))**2)
    logp_Airbag = -0.5*np.log(2*np.pi) - 0.5*((Airbag - (np.tanh(MakeModel) + 0.5*VehicleYear))**2)
    logp_GoodStudent = -0.5*np.log(2*np.pi) - 0.5*((GoodStudent - (0.6*Age - 0.4*SocioEcon))**2)
    logp_DrivingSkill = -0.5*np.log(2*np.pi) - 0.5*((DrivingSkill - (1 + np.tanh(SeniorTrain) + 0.4*Age))**2)
    logp_Cushioning = -0.5*np.log(2*np.pi) - 0.5*((Cushioning - (0.8*Airbag))**2)
    logp_DrivQuality = -0.5*np.log(2*np.pi) - 0.5*((DrivQuality - (0.7*RiskAversion + 0.6*DrivingSkill))**2)
    logp_Antilock = -0.5*np.log(2*np.pi) - 0.5*((Antilock - np.sin(VehicleYear + 0.5*MakeModel))**2)
    logp_CarValue = -0.5*np.log(2*np.pi) - 0.5*((CarValue - (np.tanh(MakeModel) + 0.4*Mileage + 0.2*VehicleYear))**2)
    logp_AntiTheft = -0.5*np.log(2*np.pi) - 0.5*((AntiTheft - (1/(1+np.exp(-SocioEcon)) + RiskAversion**2))**2)
    logp_Accident = -0.5*np.log(2*np.pi) - 0.5*((Accident - np.sin(0.5*Antilock + 0.4*Mileage + 0.4*DrivQuality))**2)
    logp_Theft = -0.5*np.log(2*np.pi) - 0.5*((Theft - (0.6*AntiTheft + 0.5*HomeBase + 0.4*CarValue))**2)
    logp_RuggedAuto = -0.5*np.log(2*np.pi) - 0.5*((RuggedAuto - (0.8*VehicleYear + 0.5*MakeModel))**2)
    logp_ThisCarDam = -0.5*np.log(2*np.pi) - 0.5*((ThisCarDam - (0.8*Accident + 0.3*RuggedAuto))**2)
    logp_OtherCar = -0.5*np.log(2*np.pi) - 0.5*((OtherCar - 0.7*SocioEcon)**2)
    logp_MedCost = -0.5*np.log(2*np.pi) - 0.5*((MedCost - (0.6*Accident + 0.5*Cushioning + np.sin(Age)))**2)
    logp_ThisCarCost = -0.5*np.log(2*np.pi) - 0.5*((ThisCarCost - (0.6*ThisCarDam + 0.5*Theft + 0.4*CarValue))**2)
    logp_OtherCarCost = -0.5*np.log(2*np.pi) - 0.5*((OtherCarCost - (0.8*Accident + 0.5*RuggedAuto))**2)
    logp_ILiCost = -0.5*np.log(2*np.pi) - 0.5*((ILiCost - (0.7*Accident))**2)
    logp_DrivHist = -0.5*np.log(2*np.pi) - 0.5*((DrivHist - (1 + np.tanh(DrivingSkill)))**2)
    logp_PropCost = -0.5*np.log(2*np.pi) - 0.5*((PropCost - (0.8*ThisCarCost + 0.5*OtherCar))**2)

    # Log-joint
    log_lik = (
        logp_Age + logp_SocioEcon + logp_RiskAversion + logp_MakeModel + logp_Mileage + logp_HomeBase +
        logp_SeniorTrain + logp_VehicleYear + logp_Airbag + logp_GoodStudent + logp_DrivingSkill +
        logp_Cushioning + logp_DrivQuality + logp_Antilock + logp_CarValue + logp_AntiTheft +
        logp_Accident + logp_Theft + logp_RuggedAuto + logp_ThisCarDam + logp_OtherCar +
        logp_MedCost + logp_ThisCarCost + logp_OtherCarCost + logp_ILiCost + logp_DrivHist + logp_PropCost
    )

    return np.sum(log_lik)