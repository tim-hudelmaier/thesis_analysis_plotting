from flask import Flask, request, jsonify
import incremental_training_experiment as ite

app = Flask(__name__)

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    config = request.get_json(force=True)
    ite.run(config)
    return jsonify([{'result': 'success'}])


@app.route('/run_single_epoch', methods=['Post'])
def run_single_epoch():
    config = request.get_json(force=True)
    ite.run_single_epoch(config)
    return jsonify([{'result': 'success'}])
