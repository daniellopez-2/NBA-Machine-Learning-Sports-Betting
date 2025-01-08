def american_to_decimal(american_odds):
    """
    Converts American odds to decimal odds (European odds).
    """
    try:
        american_odds = float(american_odds)
        if american_odds is None:
            return None
        if american_odds >= 100:
            decimal_odds = (american_odds / 100) + 1
        else:
            decimal_odds = (-100 / american_odds) + 1
        return round(decimal_odds, 2)
    except (ValueError, TypeError):
        return None

def calculate_kelly_criterion(american_odds, model_prob):
    """
    Calculates the fraction of the bankroll to be wagered on each bet
    """
    try:
        if american_odds is None or model_prob is None:
            return None
        
        decimal_odds = american_to_decimal(american_odds)
        if decimal_odds is None:
            return None
            
        bankroll_fraction = ((decimal_odds * model_prob - (1 - model_prob)) / decimal_odds) * 100
        # Cap at 25% of bankroll
        bankroll_fraction = min(max(0, bankroll_fraction), 25)
        return round(bankroll_fraction, 2)
    except:
        return None
