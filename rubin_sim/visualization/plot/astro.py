import pandas as pd
import bokeh.models
from pandas.api.types import is_datetime64_any_dtype


def make_almanac_events_table(almanac_events):
    assert False
    # If we are passed a DataSource, exctract the
    # pandas.DataFrame from it.
    if isinstance(almanac_events, bokeh.models.DataSource):
        source = almanac_events
        source_df = almanac_events.to_df()
    else:
        source_df = almanac_events.copy().reset_index()

        # Remove the timezone from the DataFrame, because bokeh
        # converts all localized datetimes to UTC, and we
        # want to show the times in different timezones.
        for column in source_df:
            if is_datetime64_any_dtype(source_df[column]):
                source_df[column] = source_df[column].dt.tz_localize(None)

        source = bokeh.models.ColumnDataSource(source_df, name="almanac_events")

    columns = []
    for column in source_df:

        if is_datetime64_any_dtype(source_df[column]):
            formatter = bokeh.models.DateFormatter(format="%Y-%m-%d %H:%M:%S")
            columns.append(
                bokeh.models.TableColumn(
                    field=column, title=column, formatter=formatter
                )
            )
        else:
            columns.append(bokeh.models.TableColumn(field=column, title=column))

    table = bokeh.models.DataTable(
        source=source, name="almanac_events_table", columns=columns
    )
    return table
