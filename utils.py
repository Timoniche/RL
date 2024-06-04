import datetime


def generate_time_id():
    current_datetime = datetime.datetime.now()
    id = current_datetime.strftime('%Y_%m_%d_%H_%M')

    return id