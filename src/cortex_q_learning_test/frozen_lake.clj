(ns cortex-q-learning-test.frozen-lake
  (:require [cortex.nn.execute :as execute]
            [cortex.optimize.sgd :refer [sgd]]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.util :as util]
            [cortex-q-learning-test.gym :as gym]))

(def num-features 16)
(def num-actions 4)
(def gamma 0.99)

(defn predict [network context observation]
  (-> (execute/run
        network
        [{:features (util/idx->one-hot observation num-features)}]
        :context context)
      first
      :target))

(defn eps-greedy-select [eps qs]
  (if (> (rand) eps)
    (first (apply max-key second (doall (map-indexed vector qs))))
    (rand-int (count qs))))

(defn transition->features
  [{:keys [prev-observation] :as transition}]
  (util/idx->one-hot prev-observation num-features))

(defn transition->target
  [{:keys [reward action done] :as transition} qs qs*]
  (let [target (if done
                  reward
                  (+ reward (* gamma (apply max qs*))))]
    (assoc qs action target)))

;; frozen lake example
(defn frozen-lake-episode
  [id nnet context opt eps]
  (let [max-step 100]
    (loop [obs (gym/reset id)
           terminal? false
           goal? false
           transitions []
           nn nnet
           opt opt
           step 0]
      (if terminal?
        {:steps (count transitions) :nnet nn :optimizer opt :goal? goal?
         :rewards (->> transitions (map :reward) (reduce +))}
        (let [qs (predict nn context obs)
              action (eps-greedy-select eps qs)
              transition (merge (gym/step id action)
                                {:action action
                                 :prev-observation obs})
              {:keys [observation done reward]} transition
              done? (or done (= max-step step))
              goal? (pos? reward)
              qs* (predict nn context observation)
              target (transition->target transition qs qs*)
              features (transition->features transition)
              train-ds [{:target target :features features}]
              {:keys [network optimizer]} (execute/train nn
                                                         train-ds
                                                         :batch-size 1
                                                         :context context
                                                         :optimizer opt)]
          (recur observation
                 done?
                 goal?
                 (conj transitions transition)
                 network
                 optimizer
                 (inc step)))))))

(defonce frozen-lake-network
  (network/linear-network
   [(layers/input num-features 1 1
                  :id :features
                  :weights {:initialization {:type :xavier}})
    (layers/linear num-actions :id :target)]))

(defn train-frozen-lake-network
  [network optimizer num-episodes]
  (let [context (execute/compute-context)
        id (gym/instance-id "FrozenLake-v0")]
    (execute/with-compute-context
      context
      (loop [nn network
             opt optimizer
             episode 0
             episode-rewards []
             episode-steps []
             successes 0]
        (if (= episode num-episodes)
          {:nn nn :rewards episode-rewards :steps episode-steps
           :mean-episode-reward (/ (reduce + episode-rewards) num-episodes)
           :mean-episode-length (/ (reduce + episode-steps) num-episodes)}
          (let [epsilon (/ 1.0 (+ 10.0 (/ (inc episode) 50.0)))
                {:keys [steps nnet optimizer goal? rewards]} (frozen-lake-episode
                                                              id nn context opt epsilon)]
            (recur nnet
                   optimizer
                   (inc episode)
                   (conj episode-rewards rewards)
                   (conj episode-steps steps)
                   (if goal? (inc successes) successes))))))))

(defn frozen-lake-example []
  (let [network frozen-lake-network
        optimizer (sgd :learning-rate 0.2 :momentum 0.0)
        num-episodes 2000
        {:keys [mean-episode-reward mean-episode-length rewards]}
        (train-frozen-lake-network network optimizer num-episodes)]
    mean-episode-reward))
