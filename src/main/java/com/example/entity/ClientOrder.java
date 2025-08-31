package com.example.entity;

import jakarta.persistence.*;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "client_order")
public class ClientOrder {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank(message = "Order number is required")
    @Size(max = 50, message = "Order number must not exceed 50 characters")
    @Column(name = "order_number", unique = true, nullable = false)
    private String orderNumber;

    @NotBlank(message = "Client ID is required")
    @Size(max = 50, message = "Client ID must not exceed 50 characters")
    @Column(name = "client_id", nullable = false)
    private String clientId;

    @NotBlank(message = "Order type is required")
    @Size(max = 20, message = "Order type must not exceed 20 characters")
    @Column(name = "order_type", nullable = false)
    private String orderType;

    @NotBlank(message = "Status is required")
    @Size(max = 20, message = "Status must not exceed 20 characters")
    @Column(name = "status", nullable = false)
    private String status = "PENDING";

    @NotNull(message = "Amount is required")
    @DecimalMin(value = "0.01", message = "Amount must be greater than 0")
    @Column(name = "amount", nullable = false, precision = 15, scale = 2)
    private BigDecimal amount;

    @Size(max = 3, message = "Currency must not exceed 3 characters")
    @Column(name = "currency", nullable = false)
    private String currency = "USD";

    @Column(name = "description", columnDefinition = "TEXT")
    private String description;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @NotBlank(message = "Created by is required")
    @Size(max = 50, message = "Created by must not exceed 50 characters")
    @Column(name = "created_by", nullable = false, updatable = false)
    private String createdBy;

    @NotBlank(message = "Updated by is required")
    @Size(max = 50, message = "Updated by must not exceed 50 characters")
    @Column(name = "updated_by", nullable = false)
    private String updatedBy;

    @Version
    @Column(name = "version", nullable = false)
    private Integer version = 1;

    // Constructors
    public ClientOrder() {}

    public ClientOrder(String orderNumber, String clientId, String orderType, BigDecimal amount, String createdBy) {
        this.orderNumber = orderNumber;
        this.clientId = clientId;
        this.orderType = orderType;
        this.amount = amount;
        this.createdBy = createdBy;
        this.updatedBy = createdBy;
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getOrderNumber() {
        return orderNumber;
    }

    public void setOrderNumber(String orderNumber) {
        this.orderNumber = orderNumber;
    }

    public String getClientId() {
        return clientId;
    }

    public void setClientId(String clientId) {
        this.clientId = clientId;
    }

    public String getOrderType() {
        return orderType;
    }

    public void setOrderType(String orderType) {
        this.orderType = orderType;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public BigDecimal getAmount() {
        return amount;
    }

    public void setAmount(BigDecimal amount) {
        this.amount = amount;
    }

    public String getCurrency() {
        return currency;
    }

    public void setCurrency(String currency) {
        this.currency = currency;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }

    public String getCreatedBy() {
        return createdBy;
    }

    public void setCreatedBy(String createdBy) {
        this.createdBy = createdBy;
    }

    public String getUpdatedBy() {
        return updatedBy;
    }

    public void setUpdatedBy(String updatedBy) {
        this.updatedBy = updatedBy;
    }

    public Integer getVersion() {
        return version;
    }

    public void setVersion(Integer version) {
        this.version = version;
    }

    @Override
    public String toString() {
        return "ClientOrder{" +
                "id=" + id +
                ", orderNumber='" + orderNumber + '\'' +
                ", clientId='" + clientId + '\'' +
                ", orderType='" + orderType + '\'' +
                ", status='" + status + '\'' +
                ", amount=" + amount +
                ", currency='" + currency + '\'' +
                ", version=" + version +
                '}';
    }
} 