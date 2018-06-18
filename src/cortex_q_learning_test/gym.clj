(ns cortex-q-learning-test.gym
  (:require [clj-http.client :as client]))

;; There are Clojure binding for the Gym http API,
;; but you have to install boot and I really don't
;; want to deal with that. This was simple enough.

(def base "http://127.0.0.1:5000/v1/envs/")

(defn instance-id [env]
  (-> (client/post base
                   {:form-params {:env_id env}
                    :content-type :json
                    :as :auto})
      :body
      :instance_id))

(defn reset [id]
  (-> (client/post (str base id "/reset/")
                   {:content-type :json :as :auto})
      :body
      :observation))

(defn step [id action]
  (-> (client/post (str base id "/step/")
                   {:form-params {:action action}
                    :content-type :json :as :auto})
      :body
      (select-keys [:done :observation :reward])))
