class Task
  include ActiveModel::Model
  include ActiveModel::Attributes

  attribute :task_id, :integer
  attribute :name, :string
  attribute :description, :string
  attribute :estimated_hours, :integer
  attribute :estimated_cost_usd, :integer
  attribute :complexity, :string
  attribute :includes, default: -> { [] }
end
