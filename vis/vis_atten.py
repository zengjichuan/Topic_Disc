import json
from flask import Flask, request, render_template, Markup
app = Flask(__name__)

@app.route("/")
def hello():
	data_file = open("./gen_samples.txt")
	data = json.load(data_file)
	colors = [		
		["#ff7700","#ff851c","#ff9c47","#ffa960","#ffc08c","#ffd4b2","#ffe2cc","#ffefe2","#fff5ed","#fffaf7"],
		["#71fc00","#8bff2d","#9cfc4e","#aeff6d","#bafc85","#cdffa5","#d8ffba","#e4ffd1","#ecfce0","#f7fff2"],
		["#00fce7","#26ffec","#53fced","#77fff3","#b2fff8","#c4fff9","#d6fffa","#e0fffb","#edfffc","#f7fffd"],
		["#00c3ff","#21cbff","#5bd8ff","#72ddff","#91e4ff","#a8e9ff","#bceeff","#cef2ff","#d8f4ff","#effaff"],
		["#0037ff","#1c4dff","#345ff9","#5177ff","#6b8bff","#7e99fc","#97adfc","#b5c5ff","#c9d4fc","#e5ebff"],
		["#6005ff","#7121ff","#813aff","#9459ff","#a372ff","#af85fc","#c3a4fc","#dcc9ff","#e5d9fc","#f2edfc"],
		["#ff02e1","#fc28e3","#fc41e6","#fc5de9","#fc79ec","#fc94ef","#fcabf1","#fcc4f4","#f9def5","#fff7fd"],
		["#ff026b","#ff217d","#fc3a8a","#fc559a","#f76aa4","#f77eb0","#f9a4c8","#fcbdd8","#fcd1e4","#fceaf2"],
		["#fffa00","#fffa19","#fffa35","#fffa56","#fffa75","#fffb91","#fffcaf","#fffdcc","#fffde0","#fffdef"],
		["#00ffe5","#1effe8","#38ffea","#51ffec","#70ffee","#89fff1","#a3fff4","#bffff7","#d6fff9","#e5fffb"],
		["#f70000","#ff2d2d","#ff4444","#ff6363","#ff7a7a","#fc8d8d","#fc9f9f","#ffc1c1","#fcd4d4","#ffeded"],
	]
	vis_data = []
	for idx, term in enumerate(data):
		texts = data[idx]["target"]
		sources = [4 if i <10 else 10 for i in data[idx]["source"]]
		weights = [i if i <1.0 else 1.0 for i in data[idx]["weight"]]
		new_colors = [colors[sources[i]][-int(round(weights[i]*10))] for i in range(len(texts))]
		text_color = zip(texts, new_colors)
		html_text = ""
		for ele in text_color:
			html_text+="<span style='background-color: "+ele[1]+"'>" + ele[0]+" </span>"
		vis_data.append(Markup(html_text))
	data_file.close()
	return render_template('vis.html', vis_data=vis_data)

if __name__ == "__main__" :
	app.run()