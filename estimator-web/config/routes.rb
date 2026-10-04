Rails.application.routes.draw do
  # Reveal health status on /up that returns 200 if the app boots with no exceptions, otherwise 500.
  get "up" => "rails/health#show", as: :rails_health_check

  resources :estimations, only: [ :index, :new, :create, :show ]
  root "estimations#new"
end
