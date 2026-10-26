"""Reusable Streamlit reservation page. Import render_reservation_page() from app.py."""
from datetime import date, time
import pandas as pd
import streamlit as st
import database as db
from recommendations import build_recommendations


def render_reservation_page():
    db.init_db()
    st.header('📅 Reservation Management')
    st.caption('Demo bookings are stored in a local SQLite database. Use sample data for the college demonstration.')

    with st.expander('➕ Add a reservation', expanded=True):
        with st.form('add_reservation_form', clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input('Customer name *')
                booking_date = st.date_input('Booking date', value=date.today(), min_value=date.today())
                booking_time = st.time_input('Booking time', value=time(19, 0), step=900)
                party_size = st.number_input('Party size', min_value=1, max_value=30, value=2)
            with c2:
                phone = st.text_input('Contact number (optional)')
                advance_days = st.number_input('Days booked in advance', min_value=0, max_value=365, value=1)
                past_visits = st.number_input('Previous visits', min_value=0, max_value=500, value=0)
                spend = st.number_input('Estimated spend (₹)', min_value=0.0, max_value=100000.0, value=600.0, step=50.0)
            weather = st.selectbox('Expected weather', ['Sunny', 'Rainy', 'Cloudy', 'Windy'])
            submitted = st.form_submit_button('Save reservation', type='primary', use_container_width=True)
            if submitted:
                if not name.strip():
                    st.error('Please enter a customer name.')
                else:
                    db.add_reservation(name, booking_date.isoformat(), booking_time.strftime('%H:%M'), party_size,
                                       phone, advance_days, past_visits, spend, weather)
                    st.success('Reservation saved.')
                    st.rerun()

    df = db.get_reservations()
    st.subheader('Saved reservations')
    if df.empty:
        st.info('No reservations yet. Add one above or load demo data from the Dashboard.')
        return

    display_cols = ['id', 'customer_name', 'booking_date', 'booking_time', 'party_size', 'estimated_spend', 'weather', 'status']
    st.dataframe(df[display_cols].rename(columns={
        'id': 'Booking ID', 'customer_name': 'Customer', 'booking_date': 'Date', 'booking_time': 'Time',
        'party_size': 'Guests', 'estimated_spend': 'Estimated spend (₹)', 'weather': 'Weather', 'status': 'Status'
    }), use_container_width=True, hide_index=True)

    with st.expander('Update booking status or delete a booking'):
        ids = df['id'].astype(int).tolist()
        chosen = st.selectbox('Choose booking ID', ids, format_func=lambda x: f"#{x} — {df.loc[df['id'] == x, 'customer_name'].iloc[0]}")
        c1, c2 = st.columns(2)
        with c1:
            status = st.selectbox('New status', ['Booked', 'Confirmed', 'Arrived', 'Completed', 'Cancelled', 'No-show'])
            if st.button('Update status', use_container_width=True):
                db.update_status(chosen, status)
                st.success('Status updated.')
                st.rerun()
        with c2:
            confirm_delete = st.checkbox('I understand this permanently deletes the booking')
            if st.button('Delete booking', disabled=not confirm_delete, use_container_width=True):
                db.delete_reservation(chosen)
                st.success('Booking deleted.')
                st.rerun()

    st.subheader('Operational recommendations')
    for tip in build_recommendations(df):
        st.info(tip)

    st.download_button('⬇️ Download reservations CSV', df.to_csv(index=False).encode('utf-8'),
                       file_name='cafe_insight_reservations.csv', mime='text/csv', use_container_width=True)
