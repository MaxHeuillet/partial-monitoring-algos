# fig.add_traces([ go.Scatter( x=dft.index, y=[ np.sqrt(x) for x in sequence ], name= r'$\frac{1}{n}$' ) ])


# fig.update_layout(xaxis=dict(  title='Input Sequence', gridcolor='white', gridwidth=0.75, ),
#                   yaxis=dict(  title='Cumulative Regret', gridcolor='white',  gridwidth=0.75,),
#                   paper_bgcolor='rgb(243, 243, 243)',
#                   plot_bgcolor='rgb(243, 243, 243)',
#                   legend=dict(yanchor="top",y=1.1,xanchor="left",x=0.01) )