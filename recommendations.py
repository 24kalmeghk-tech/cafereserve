"""Transparent rule-based recommendations for the Cafe Insight demonstration."""
from datetime import date
import pandas as pd

def build_recommendations(reservations: pd.DataFrame, predicted_no_show_rate=None):
    tips = []
    if reservations is None or reservations.empty:
        return ["Add a few reservations to generate café operations recommendations."]

    active = reservations[~reservations['status'].isin(['Cancelled', 'No-show'])].copy()
    if active.empty:
        return ["There are no active reservations. Review cancelled bookings or add upcoming reservations."]

    today = date.today().isoformat()
    today_rows = active[active['booking_date'].astype(str) == today]
    if not today_rows.empty:
        count = len(today_rows)
        guests = int(today_rows['party_size'].sum())
        tips.append(f"Today: plan for about {guests} guests across {count} reservations. Check table capacity before confirming walk-ins.")
        hour_counts = today_rows.copy()
        hour_counts['hour'] = hour_counts['booking_time'].astype(str).str[:2]
        busiest = hour_counts.groupby('hour').size().sort_values(ascending=False)
        if not busiest.empty:
            hour = busiest.index[0]
            tips.append(f"Your busiest booked hour today is around {hour}:00. Review floor and kitchen coverage for this period.")
    else:
        tips.append("No bookings are saved for today. Add real bookings or use the demo dataset to test the dashboard.")

    at_risk = active[(active['advance_days'] >= 7) & (active['past_visits'] == 0)]
    if not at_risk.empty:
        tips.append(f"{len(at_risk)} booking(s) have a simple demo risk flag (booked well ahead with no past visits). Consider sending a friendly confirmation reminder; this is not a validated ML prediction.")
    else:
        tips.append("No reservations match the demo reminder rule. Keep using confirmation messages for large or important bookings.")

    total_spend = float(active['estimated_spend'].sum())
    tips.append(f"Estimated spend across active saved bookings is ₹{total_spend:,.0f}. This is an estimate from entered values, not recorded revenue.")
    tips.append("For food preparation, use actual item-level sales before making quantity recommendations; this prototype does not yet track menu inventory.")
    return tips
