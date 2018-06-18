(ns cortex-q-learning-test.cliffwalking
  (:require [cortex.nn.execute :as execute]
            [cortex.optimize.sgd :refer [sgd]]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.util :as util]))

(def cliffwalking-gridworld
  {:start 36
   :goal 47
   :actions {:up 0 :down 1 :right 2 :left 3}
   :top-edge #{0 1 2 3 4 5 6 7 8 9 10 11}
   :bottom-edge #{36 37 38 39 40 41 42 43 44 45 46 47}
   :right-edge #{11 23 35 47}
   :left-edge #{0 12 24 36}
   :cliff #{37 38 39 40 41 42 43 44 45 46}})

(defn grid-step [state action]
  (let [{:keys [top-edge bottom-edge left-edge right-edge
                actions start cliff goal]} cliffwalking-gridworld
        {:keys [up down left right]} actions
        next-state (cond
                     ;; attempted moves off grid don't change state
                     (or (and (= action up) (top-edge state))
                         (and (= action down) (bottom-edge state))
                         (and (= action right) (right-edge state))
                         (and (= action left) (left-edge state))) state
                     (= action up) (- state 12)
                     (= action down) (+ state 12)
                     (= action right) (inc state)
                     (= action left) (dec state))
        cliff? (some #(= next-state %) cliff)
        goal? (= next-state goal)]
    {:action action :state state
     :reward (if cliff? -100.0 -1.0)
     :next-state next-state
     :goal-state? goal?
     :terminal-state? (or cliff? goal?)}))

(def num-features 48)
(def num-actions 4)
(def gamma 0.99)

(defn predict [network context state]
  (-> (execute/run
        network
        [{:features (util/idx->one-hot state num-features)}]
        :context context)
      first
      :target))

(defn eps-greedy-select [eps qs]
  (if (> (rand) eps)
    (first (apply max-key second (doall (map-indexed vector qs))))
    (rand-int (count qs))))

(defn transition->features
  [{:keys [state] :as transition}]
  (util/idx->one-hot state num-features))

(defn transition->target
  [{:keys [reward action terminal-state?] :as transition} qs qs*]
  (let [target (if terminal-state?
                  reward
                  (+ reward (* gamma (apply max qs*))))]
    (assoc qs action target)))

(defn cliffwalking-episode
  [nnet context opt eps]
  (loop [state (:start cliffwalking-gridworld)
           terminal? false
           goal? false
           transitions []
           nn nnet
           opt opt
           step 0]
      (if terminal?
        {:steps (count transitions) :nnet nn :optimizer opt :goal? goal?
         :rewards (->> transitions (map :reward) (reduce +))}
        (let [qs (predict nn context state)
              action (eps-greedy-select eps qs)
              transition (grid-step state action)
              {:keys [next-state terminal-state? goal-state?]} transition
              qs* (predict nn context next-state)
              features (transition->features transition)
              target (transition->target transition qs qs*)
              train-ds [{:target target :features features}]
              {:keys [network optimizer]} (execute/train nn
                                                         train-ds
                                                         :context context
                                                         :batch-size 1
                                                         :optimizer opt)]
          (recur next-state
                 terminal-state?
                 goal-state?
                 (conj transitions transition)
                 network
                 optimizer
                 (inc step))))))

(defonce cliffwalking-network
  (network/linear-network
   [(layers/input num-features 1 1
                  :id :features
                  :weights {:initialization {:type :xavier}})
    (layers/linear num-actions :id :target)]))

(defn train-cliffwalking-network
  [network optimizer num-episodes]
  (let [context (execute/compute-context)]
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
          (let [epsilon 0.05
                {:keys [steps nnet optimizer goal? rewards]} (cliffwalking-episode
                                                              nn context opt epsilon)]
            (recur nnet
                   optimizer
                   (inc episode)
                   (conj episode-rewards rewards)
                   (conj episode-steps steps)
                   (if goal? (inc successes) successes))))))))

(defn cliffwalking-example []
  (let [network cliffwalking-network
        optimizer (sgd :learning-rate 0.1 :momentum 0.0)
        num-episodes 200
        {:keys [mean-episode-reward mean-episode-length rewards]}
        (train-cliffwalking-network network optimizer num-episodes)]
    (take-last 20 rewards)))
