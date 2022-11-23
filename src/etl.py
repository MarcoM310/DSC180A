def get_data(fp, years, states):
    data = pd.read_csv(fp)
    data = (data
            .loc[df['year'].isin(years)]
			  .loc[df['loc'].isin(states)])
    return data
