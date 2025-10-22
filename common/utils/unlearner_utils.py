from common.unlearners.unlearners import (
    RetrainUnlearner, GDUnlearner, GAUnlearner, NGDUnlearner, NGPUnlearner,
    ScrubUnlearner, NPOUnlearner, SalientUnlearner, L1SparseUnlearner, RidgeUnlearner, MinNormOGUnlearner
)

VALID_ALGS=["GD","GA","MinNormOG","NGD","NGP","NPO","Ridge","Scrub","SalUn","L1Sparse","Retrain"]

def unlearner_from_alg(alg, model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier):
    if alg == "Retrain":
        return RetrainUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "GD":
        return GDUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "GA":
        return GAUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "NGD":
        return NGDUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "NGP":
        return NGPUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "Scrub":
        return ScrubUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "NPO":
        return NPOUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "SalUn":
        return SalientUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "L1Sparse":
        return L1SparseUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "Ridge":
        return RidgeUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    elif alg == "MinNormOG":
        return MinNormOGUnlearner(model, optimizer, retain_loader, forget_loader, logger, cfg, is_classifier)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")